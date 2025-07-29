import torch
from transformers import AutoModel, AutoTokenizer
import string
import os
import cv2
import json
import numpy as np
import torchvision.transforms as T
import datetime
from torchvision.transforms.functional import InterpolationMode
import concurrent.futures
from collections import defaultdict
import imageio
import multiprocessing
import gymnasium as gym
import mani_skill.envs
from PIL import Image
import traceback
import shutil
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from typing import List



def action_to_str(action, num_floats: int = 4):
    return [np.round(a, num_floats) for a in action.values()] if isinstance(action, dict) else [np.round(a, num_floats) for a in action]


def video_writing(frames: List[List[np.ndarray]], save_path: str, fps: int = 10):
    max_length = max([len(frame) for frame in frames])
    num_in_one = len(frames)
    # concatenate all the 16 frames of the same timestep into one frame (4*4)
    num_per_side = np.sqrt(num_in_one).astype(int)
    frames_concat = []
    for i in range(max_length):
        frame = []
        for j in range(len(frames)):
            if i < len(frames[j]):
                frame.append(frames[j][i])
            elif len(frames[j]) == 0:
                # if the frame is empty, fill with a black frame
                frame.append(np.zeros_like(frames[0][0]))
            else:
                # if the frame is not enough, fill with the last frame, with word "Terminal" on it 
                last_frame = frames[j][-1]
                frame.append(write_terminal(last_frame, "Terminated"))
                
        # concatenate the frames in the same timestep into one frame (4 * 4)
        rows = [np.concatenate(frame[i * num_per_side:(i + 1) * num_per_side], axis=1) for i in range(num_per_side)]
        if len(rows) == 1:
            frame_concat = rows[0]
        else:
            frame_concat = np.concatenate(rows, axis=0)
        frames_concat.append(frame_concat)
    with imageio.get_writer(save_path, fps=fps, ffmpeg_params=['-loglevel', 'error']) as writer:
        for frame in frames_concat:
            writer.append_data(frame)


def write_instruction_action(instruction: str, rgb: np.ndarray, action: str = None, raw_action: str = None):
    """
    在图片上方增加一个白色背景条，并写入 instruction。
    如果提供了 action 参数，则会在 instruction 下方额外写入一行 action。

    :param instruction: 要显示的第一行指令文本。
    :param rgb: 输入的原始图像 (numpy array)。
    :param action: (可选) 要在第二行显示的动作文本。
    :return: 带有文本的新图像。
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_color = (0, 0, 0)  # 黑色字体
    bg_color = (255, 255, 255)  # 白色背景
    
    # --- 动态计算所需空间 ---
    texts_to_draw = [instruction]
    if action is not None:
        texts_to_draw.append(action)
    if raw_action is not None:
        texts_to_draw.append(raw_action)

    # 获取每行文本的尺寸
    text_sizes = [cv2.getTextSize(text, font, font_scale, font_thickness)[0] for text in texts_to_draw]
    text_heights = [size[1] for size in text_sizes]

    # 定义边距和行间距
    top_margin = 10
    bottom_margin = 10
    line_spacing = 5 # 两行文字之间的额外间距

    # 计算总的 padding 高度
    total_text_height = sum(text_heights)
    if len(texts_to_draw) > 1:
        total_text_height += line_spacing * (len(texts_to_draw) - 1)
    
    pad_top = total_text_height + top_margin + bottom_margin

    # --- 创建并绘制新图像 ---
    h, w, _ = rgb.shape
    new_img = np.full((h + pad_top, w, 3), bg_color, dtype=np.uint8)

    # 把原图粘贴到新图像的下方
    new_img[pad_top:, :] = rgb

    # --- 逐行写入文本 ---
    current_y = top_margin
    for i, text in enumerate(texts_to_draw):
        text_h = text_heights[i]
        # 计算文本基线的 y 坐标 (putText 的 y 坐标是基线位置)
        text_y = current_y + text_h
        text_x = 10  # 左边距

        cv2.putText(new_img, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        
        # 更新下一行文本的起始 y 坐标
        current_y = text_y + line_spacing

    return new_img

def write_terminal(frame, word):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 4
    text_color = (255, 0, 0)  # 红色字体
    # 获取原始图片尺寸
    h, w, _ = frame.shape

    # 计算文本大小
    (text_width, text_height), _ = cv2.getTextSize(word, font, font_scale, font_thickness)

    # 设置文本位置为中心
    text_x = (w - text_width) // 2
    text_y = (h + text_height) // 2

    # 在图像上写入文本
    cv2.putText(frame, word, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return frame

class VideoRecorder:
    def __init__(self, save_path, fps=20, num_envs=1):
        self.save_path = save_path
        self.fps = fps
        self.env_id_num = defaultdict(int)
        self.recorder = defaultdict(list)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)
        self.futures = []
        self.venv_reward = defaultdict(float)
        self.num_envs = num_envs
        self.done_nums = 0
        os.makedirs(self.save_path, exist_ok=True)

        
    def _check_futures(self):
        done, _ = concurrent.futures.wait(self.futures, timeout=0)
        for future in done:
            try:
                future.result()
            except Exception as e:
                print(f"Video save failed: {e}")
        self.futures = [f for f in self.futures if not f.done()]        

    def append_obs(self, cameras: List[np.ndarray], rewards: List[float], terminated_status: np.ndarray, actions=None, raw_actions=None):
        if actions is None:
            actions = [None] * self.num_envs
        if raw_actions is None:
            raw_actions = [None] * self.num_envs
        metrics = {}
        for env_id in range(self.num_envs):
            reward = rewards[env_id]
            is_terminated = terminated_status[env_id]
            action = actions[env_id]
            raw_action = raw_actions[env_id]
            camera = [cam[env_id] for cam in cameras]
            camera = camera[0] if len(camera) == 1 else np.concatenate(camera, axis=1)    
            # print(camera.shape, "camera")
            self.venv_reward[env_id] += reward
            # print(reward)
            instruction = f"Rew: {reward} EPR: {self.venv_reward[env_id]}"
            raw_action = f"RA: {action_to_str(raw_action, 3)}" if raw_action is not None else None
            action = f"A: {action_to_str(action, 3)}" if action is not None else None
            camera = write_instruction_action(instruction, camera, action, raw_action)
            # print(camera.shape, "inst")
            self.recorder[env_id].append(camera)       
            if is_terminated:
                file_name = f"{self.done_nums}_{env_id}_{self.env_id_num[env_id]}_{len(self.recorder[env_id])}_{self.venv_reward[env_id]}.mp4"
                save_path = f"{self.save_path}/{file_name}"
                print(f"Writing video: {save_path}, Num steps: {len(self.recorder[env_id])}, Reward: {self.venv_reward[env_id]}")
                future = self.executor.submit(
                    video_writing, 
                    [self.recorder[env_id]], 
                    save_path, 
                    fps=self.fps
                )
                self.futures.append(future)
                self.env_id_num[env_id] += 1
                metrics[self.done_nums] = {
                    "env_id": env_id,
                    'num_steps': len(self.recorder[env_id]),
                    "sparse_reward": self.venv_reward[env_id],
                    "file_name": file_name
                }    
                self.recorder[env_id] = []    
                self.venv_reward[env_id] = 0
                self.done_nums += 1
        self._check_futures()  # 每次添加新任务后检查已完成的任务
        return metrics
          

    # def append_obs(self, observation, rewards, terminated_status, actions=None, raw_actions=None):
    #     if actions is None:
    #         actions = [None] * self.num_envs
    #     if raw_actions is None:
    #         raw_actions = [None] * self.num_envs
    #     metrics = {}
    #     for env_id in range(self.num_envs):
    #         reward = rewards[env_id]
    #         is_terminated = terminated_status[env_id]
    #         action = actions[env_id]
    #         raw_action = raw_actions[env_id]
    #         camera = observation['sensor_data']["base_camera"]["rgb"][env_id].cpu().numpy()
    #         # print(camera.shape, "camera")
    #         self.venv_reward[env_id] += reward
    #         # print(reward)
    #         instruction = f"Rew: {reward} EPR: {self.venv_reward[env_id]}"
    #         raw_action = f"RA: {action_to_str(raw_action, 3)}" if raw_action is not None else None
    #         action = f"A: {action_to_str(action, 3)}" if action is not None else None
    #         camera = write_instruction_action(instruction, camera, action, raw_action)
    #         # print(camera.shape, "inst")
    #         self.recorder[env_id].append(camera)       
    #         if is_terminated:
    #             file_name = f"{self.done_nums}_{env_id}_{self.env_id_num[env_id]}_{len(self.recorder[env_id])}_{self.venv_reward[env_id]}.mp4"
    #             save_path = f"{self.save_path}/{file_name}"
    #             print(f"Writing video: {save_path}, Num steps: {len(self.recorder[env_id])}, Reward: {self.venv_reward[env_id]}")
    #             future = self.executor.submit(
    #                 video_writing, 
    #                 [self.recorder[env_id]], 
    #                 save_path, 
    #                 fps=self.fps
    #             )
    #             self.futures.append(future)
    #             self.env_id_num[env_id] += 1
    #             metrics[self.done_nums] = {
    #                 "env_id": env_id,
    #                 'num_steps': len(self.recorder[env_id]),
    #                 "sparse_reward": self.venv_reward[env_id],
    #                 "file_name": file_name
    #             }    
    #             self.recorder[env_id] = []    
    #             self.venv_reward[env_id] = 0
    #             self.done_nums += 1
    #     self._check_futures()  # 每次添加新任务后检查已完成的任务
    #     return metrics
                
    def close(self):
        print("Closing VideoRecorder, waiting for pending video writes to finish...")
        # shutdown(wait=True) 会阻止新任务提交，并等待所有已提交任务完成
        self.executor.shutdown(wait=True)
        # 最后再检查一次，确保捕获所有任务的异常
        self._check_futures()
        print("All video writing tasks are complete.")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    if isinstance(image_file, str):
        image = Image.open(image_file).convert('RGB')
    elif isinstance(image_file, np.ndarray):
        image = Image.fromarray(image_file)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def read_jsonl_standard(file_path: str) -> list:
    """
    使用 Python 标准库逐行读取 JSONL 文件。
    
    :param file_path: JSONL 文件的路径。
    :return: 一个包含所有解析后的JSON对象（字典）的列表。
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 移除行尾可能存在的空白字符（包括换行符）
                clean_line = line.strip()
                if clean_line:  # 确保不是空行
                    # 解析当前行
                    data.append(json.loads(clean_line))
    except FileNotFoundError:
        print(f"错误：文件未找到于 '{file_path}'")
    except json.JSONDecodeError as e:
        print(f"错误：文件 '{file_path}' 中存在JSON解析错误: {e}")
    
    return data

def append_to_jsonl(new_data, filename='log.jsonl'):
    """向JSON Lines文件追加一条新记录。"""
    with open(filename, 'a', encoding='utf-8') as f:
        # 将字典转换为JSON字符串，并在末尾添加换行符
        f.write(json.dumps(new_data, ensure_ascii=False) + '\n')
        
def parse_and_validate_vector(input_str: str):
    """
    解析并验证一个字符串，期望其格式为包含7个空格分隔的数字的花括号包围的向量。

    Args:
        input_str: 模型的原始输出字符串。

    Returns:
        如果格式完全正确，则返回一个包含7个整数的列表。
        如果格式有任何问题（缺少花括号、数字数量不对、包含非数字内容等），则返回 None。
    """
    # 1. 基础检查：确保输入是字符串
    if not isinstance(input_str, str):
        return None

    # 2. 预处理：去除首尾多余的空白字符
    s = input_str.strip()

    # 3. 验证格式：是否被花括号包围
    if not (s.startswith('{') and s.endswith('}')):
        return None

    # 4. 提取花括号内的内容
    content = s[1:-1].strip()
    
    # 如果内容为空（例如输入是 "{}" 或 "{ }"），也视为无效
    if not content:
        return None

    # 5. 分割内容
    parts = content.split()

    # 6. 验证数量：是否正好是7个数字
    if len(parts) != 7:
        return None

    # 7. 验证内容：尝试将所有部分转换为整数
    try:
        vector = [int(p) for p in parts]
        return np.array(vector, dtype=np.float32)  # 返回一个整数类型的NumPy数组
    except ValueError:
        # 如果任何一部分无法转换为整数（例如 "1.5", "abc"），则捕获异常
        print("Error response:", input_str)
        return None


class InternVLEvalAgent:
    def __init__(
        self, 
        model_path,
        instruction=None,
        parent_tag: str = None,
        inference_tag: str = None,
        num_envs: int = 1,
        device: str = "cuda:0"
    ):
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        modeling_name = "modeling_internvl_chat.py"
        modeling_path = os.path.join(current_file_path, modeling_name)
        modeling_to_replace = os.path.join(model_path, "modeling_internvl_chat.py")
        if os.path.islink(modeling_to_replace):
            os.remove(modeling_to_replace)
        elif not os.path.exists(modeling_to_replace):
            # If the modeling file does not exist, copy it from the current directory
            shutil.copy(src=modeling_path, dst=modeling_to_replace)
        elif os.path.islink(modeling_to_replace) or (os.path.exists(modeling_to_replace) and os.path.exists(modeling_to_replace + ".bak")):
            os.remove(modeling_to_replace)
            shutil.copy(src=modeling_path, dst=modeling_to_replace)
        elif os.path.exists(modeling_to_replace) and not os.path.exists(modeling_to_replace + ".bak"):
            os.rename(modeling_to_replace, modeling_to_replace + ".bak")
            os.symlink(modeling_path, modeling_to_replace)
        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        self.generation_config = dict(max_new_tokens=16, do_sample=True)
        self.instruction = "stack all the cubes" if instruction is None else instruction
        jsonl_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if inference_tag is None else inference_tag
        self.jsonl_path = os.path.join(model_path, parent_tag, jsonl_name, 'inference.jsonl')
        self.num_envs = num_envs
        self.device = device
        if "dual" in model_path:
            self.dual_cam = True
        else:
            self.dual_cam = False
        # if "_noState" in model_path:
        #     self.no_state = True
        # else:
        #     self.no_state = False
        # if '_absQ' in model_path:
        #     self.action_type = ActionType.ABS_JOINT
        # elif "_absEEF" in model_path:
        #     self.action_type = ActionType.ABS_EEF
        # elif "_deltaQ" in model_path:
        #     self.action_type = ActionType.DELTA_JOINT
        # else:
        #     self.action_type = ActionType.DELTA_EEF
        # if "_quatEEF" in model_path:
        #     self.quatEEF = True
        # else:
        #     self.quatEEF = False
        # self.inference_log = []
        self.model_path = model_path


    def get_next_action(self, observations):
        num_patches_list = []
        pixel_values = []
        questions = []
        qposes = []
        for env_id in range(self.num_envs):
            qpos = observations["agent"]["qpos"][env_id].cpu().numpy()
            qposes.append(qpos)
            camera = observations['sensor_data']["base_camera"]["rgb"][env_id].cpu().numpy()
            rescaled_qpos = np.round(qpos * 1000).astype(np.int32)
            query = f"The current position state of the robotic arm's end gripper is as follows: {{Joint_0: {rescaled_qpos[0]}, Joint_1: {rescaled_qpos[1]}, Joint_2: {rescaled_qpos[2]}, Joint_3: {rescaled_qpos[3]}, Joint_4: {rescaled_qpos[4]}, Joint_5: {rescaled_qpos[5]}, Joint_6: {rescaled_qpos[6]}, Joint_7: {rescaled_qpos[7]}, Joint_8: {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            if self.dual_cam:
                query = "<image><image>" + query
            else:
                query = "<image>" + query
            pixel_0 = load_image(camera, max_num=12).to(torch.bfloat16).to(self.device)
            patch_list = []
            pixels = []
            patch_list.append(pixel_0.size(0))
            pixels.append(pixel_0)
            if self.dual_cam:
                hand_camera = observations['sensor_data']["hand_camera"]["rgb"][env_id].cpu().numpy()
                pixel_1 = load_image(hand_camera, max_num=12).to(torch.bfloat16).to(self.device)
                patch_list.append(pixel_1.size(0))
                pixels.append(pixel_1)
            if len(pixels) == 1:
                pixels = pixels[0]
            else:
                pixels = torch.cat(pixels, dim=0)
            questions.append(query)
            pixel_values.append(pixels)
            num_patches_list.append(patch_list)
        pixel_values = torch.cat(pixel_values, dim=0)
        responses = self.model.batch_chat_multi_img(
            self.tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=self.generation_config
        )
        
        actions = []
        for env_id in range(self.num_envs):
            question = questions[env_id]
            response = responses[env_id]
            action_extracted = parse_and_validate_vector(response)
            if action_extracted is None:
                action_extracted = np.zeros(7, dtype=np.float64)
                if qposes[env_id][-1] >= 0.037:
                    action_extracted[-1] = 1
                else:
                    action_extracted[-1] = -1
            else:
                action_extracted[:-1] = action_extracted[:-1] / 1000
            # print("-----------------------------------------------------")
            action_to_print = [np.round(a, 3) for a in action_extracted.values()] if isinstance(action_extracted, dict) else [np.round(a, 3) for a in action_extracted]
            # print(f'User: {question}\nAction: {action_to_print}')
            # print("-----------------------------------------------------")
            actions.append(action_extracted)
            append_to_jsonl({
                "question": question,
                "response": response,
                "action_vector": [float(a) for a in action_to_print],
                'qpos': [float(np.round(q, 3)) for q in qposes[env_id]],
                'env_id': env_id,
            }, self.jsonl_path)
        return np.array(actions)
    
def eval_checkpoint(model_parent, ckpt_name, gpu_id):
    # --- Key Change 3: Set the GPU for this specific process ---
    # This MUST be the first thing you do before any CUDA/gym/torch initialization.
    print(f"Process {os.getpid()} starting evaluation of {ckpt_name} on GPU {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Assuming your agent and env setup use the GPU
    # from your_agent_file import InternVLEvalAgent

    inference_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    num_envs = 32
    model_path = os.path.join(model_parent, ckpt_name)
    
    # Wrap the core logic in a try...finally block to ensure cleanup
    parent_tag = "mani_infer"
    max_episode_steps = 200
    eval_steps = 400
    try:
        # It's good practice to pass the device to your agent
        # The agent should then use this device, e.g., 'cuda:0'
        # Note: After setting CUDA_VISIBLE_DEVICES, GPU 0 for this process *is* the assigned gpu_id
        agent = InternVLEvalAgent(
            model_path=model_path,
            instruction="stack all the cubes",
            parent_tag=parent_tag,
            inference_tag=inference_tag,
            num_envs=num_envs,
            device=f"cuda:0" 
        )
        
        # It's good practice to include the GPU ID in the save path
        save_dir = os.path.join(model_path, parent_tag, f"{inference_tag}")
        video_recorder = VideoRecorder(save_path=os.path.join(save_dir, "videos"), fps=30, num_envs=num_envs)
        
        eval_envs = gym.make(
            "StackCube-v1",
            num_envs=num_envs,
            obs_mode="rgb",
            control_mode="pd_ee_delta_pose",
            sensor_configs={'height': 480, 'width': 480},
            max_episode_steps=max_episode_steps,
            reconfiguration_freq=1,
            reward_mode='sparse',
        )
        eval_envs = ManiSkillVectorEnv(eval_envs, auto_reset=True, ignore_terminations=True, record_metrics=True)
        metrics = {}
        obs, _ = eval_envs.reset(seed=0)
        eval_metrics = defaultdict(list)
        last_obs = obs
        for i in range(eval_steps):
            actions = agent.get_next_action(obs)
            obs, reward, terminated, truncated, infos = eval_envs.step(actions)
            # success = infos['success'].cpu().numpy()
            reward = reward.cpu().numpy()
            is_terminals = np.logical_or(truncated.cpu().numpy(), terminated.cpu().numpy())
            cameras = [last_obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), last_obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()]
            video_recorder.append_obs(cameras, reward, is_terminals, actions)
            last_obs = obs
            done_env_ids = np.where(is_terminals)[0]
            for env_id in done_env_ids:
                print(f"GPU {gpu_id}: Env {env_id} terminated, reward: {reward[env_id]}.")
            num_dones -= len(done_env_ids)
            if truncated.any():
                for k, v in infos["final_info"]["episode"].items():
                    eval_metrics[k].append(v.float())
        
        video_recorder.close()
        print(f"GPU {gpu_id}: All videos for {ckpt_name} saved to {video_recorder.save_path}.")
        for k in eval_metrics.keys():
            metrics[f"{k}_mean"] = torch.mean(torch.stack(eval_metrics[k])).item()
            eval_metrics[k] = torch.stack(eval_metrics[k]).cpu().numpy().tolist()
        metrics.update(eval_metrics)
        with open(os.path.join(save_dir, "metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=4)
            
    except Exception as e:
        print(f"GPU {gpu_id}: An error occurred while evaluating {ckpt_name}: {e}")
        traceback.print_exc()
    finally:
        if eval_envs is not None:
            eval_envs.close()
        print(f"GPU {gpu_id}: Evaluation completed for {ckpt_name}.")
        
        
if __name__ == "__main__":    
    # Set the start method for multiprocessing (important for CUDA)
    multiprocessing.set_start_method("spawn", force=True)
    
    # Define which GPUs to use
    AVAILABLE_GPUS = [0, 4, 5, 6, 7] # Modify this to match your system
    NUM_GPUS = len(AVAILABLE_GPUS)
    model_parents = [
        "vlav-project/maniskill_stack_cubes_dual/internvl2-2b/v0-20250725-182532",
        "vlav-project/maniskill_stack_cubes/internvl2-2b/v0-20250725-171104",]
    # model_parent = "/root/workspace/vlav-project/maniskill_stack_cubes_dual/internvl2-2b/v0-20250725-182532"
    # model_parent = "/root/workspace/vlav-project/maniskill_stack_cubes/internvl2-2b/v0-20250725-171104"
    for model_parent in model_parents:
        print("Model_path:", model_parent)
        # 1. Collect all tasks to be run
        tasks_to_run = []
        checkpoints = [ckpt for ckpt in os.listdir(model_parent) if ckpt.startswith("checkpoint")]
        
        for i, ckpt_name in enumerate(checkpoints):
            gpu_id = AVAILABLE_GPUS[i % NUM_GPUS] # Cycle through available GPUs
            tasks_to_run.append((model_parent, ckpt_name, gpu_id))

        print(f"Found {len(tasks_to_run)} checkpoints to evaluate on {NUM_GPUS} GPUs.")
        
        # 2. Create a process pool and run the tasks in parallel
        with multiprocessing.Pool(processes=NUM_GPUS) as pool:
            # Use starmap to pass multiple arguments to the worker function
            pool.starmap(eval_checkpoint, tasks_to_run)

        print("All evaluation tasks have been completed.")