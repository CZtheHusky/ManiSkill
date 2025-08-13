from typing import Union
import h5py
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from mani_skill.utils.io_utils import load_json
from mani_skill.utils import common
from scipy.spatial.transform import Rotation as R
import os
import cv2
import json
import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import concurrent.futures
from collections import defaultdict
import imageio
import mani_skill.envs
from PIL import Image
from typing import List
import re
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, LogitsProcessor
from copy import deepcopy
import math

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
        
import re
import numpy as np

def parse_action_vectors(s: str):
    results = []
    segments = s.strip().split('|')
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        try:
            # 用正则匹配形如 "+0 -8 -2 +3 -4 +13 +1" 的 7 个有符号整数
            matches = re.findall(r'[+-]?\d+', seg)
            if len(matches) != 7:
                results.append(None)
            else:
                results.append(np.array([int(m) for m in matches]))
        except:
            results.append(None)
    return results

def extract_action_vectors(s, vector_length, expected_count=None):
    """
    从字符串 s 中提取动作向量。
    
    参数:
        s (str): 包含若干用 {} 括起来的动作向量字符串。
        vector_length (int): 每个动作向量应包含的整数个数。
        expected_count (int, optional): 期望动作数量，如果提供且不匹配则返回 None。
        
    返回:
        如果 expected_count 不为 None 且与实际提取的动作数不符，则返回 None；
        否则返回长度为提取动作数的列表，列表中每个元素要么是长度为 vector_length 的整数列表，要么是 None（表示该动作格式不合规）。
    """
    contents = re.findall(r'\{([^}]*)\}', s)
    pattern = re.compile(r'^-?\d+(?:\s+-?\d+){' + str(vector_length - 1) + r'}$')
    results = []
    for content in contents:
        raw = content.strip()
        if pattern.match(raw):
            nums = list(map(int, raw.split()))
            results.append(np.array(nums))
        else:
            print(f"Invalid action format: '{raw}'")
            break
    if len(results) > expected_count:
        results = results[:expected_count]
    if len(results) == 0:
        return None
    return results


def parse_and_validate_vector(input_str: str, num_len: int = 7):
    if not isinstance(input_str, str):
        return None
    s = input_str.strip()
    if s.startswith('{'):
        s = s[1:]
    if s.endswith("}"):
        s = s[:-1]
    content = s
    if not content:
        return None
    parts = content.split()
    if len(parts) != num_len:
        return None
    try:
        vector = [int(p) for p in parts]
        return np.array(vector, dtype=np.float32)  
    except ValueError:
        print("Error response:", input_str)
        return None




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

def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out

def to_tensors(x, device=None):
    """
    Converts numpy arrays or dicts of numpy arrays to torch tensors.
    If device is specified, moves the tensors to that device.
    """
    if isinstance(x, dict):
        return {k: to_tensors(v, device) for k, v in x.items()}
    elif isinstance(x, np.ndarray) and device is not None:
        tensor = torch.as_tensor(x).to(device)
        return tensor
    else:
        return x
    
def action_to_str(action, num_floats: int = 4):
    return [np.round(a, num_floats) for a in action.values()] if isinstance(action, dict) else [np.round(a, num_floats) for a in action]


def quat_to_rpy(quaternion, degrees: bool = True):
    rotation_object = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rpy = rotation_object.as_euler('xyz', degrees=degrees)
    return rpy

class ManiSkillTrajectoryDataset(Dataset):
    """
    A general torch Dataset you can drop in and use immediately with just about any trajectory .h5 data generated from ManiSkill.
    This class simply is a simple starter code to load trajectory data easily, but does not do any data transformation or anything
    advanced. We recommend you to copy this code directly and modify it for more advanced use cases

    Args:
        dataset_file (str): path to the .h5 file containing the data you want to load
        load_count (int): the number of trajectories from the dataset to load into memory. If -1, will load all into memory
        success_only (bool): whether to skip trajectories that are not successful in the end. Default is false
        device: The location to save data to. If None will store as numpy (the default), otherwise will move data to that device
    """

    def __init__(self, dataset_file: str, load_count=-1, success_only: bool = False, device = None, is_episode_dataset=True) -> None:
        self.dataset_file = dataset_file
        self.device = device
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        if isinstance(load_count, int):
            if is_episode_dataset:
                self.load_dataset_episode(load_count, success_only)
            else:
                self.load_dataset(load_count, success_only, device)
        else:
            pass    
        self.is_episode_dataset = is_episode_dataset    
        
    def load_dataset_episode(self, load_count, success_only):
        self.obs = []
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only: 
                assert "success" in eps, "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                if not eps["success"]:
                    continue
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])
            
            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            self.obs.append(obs)

            self.actions.append(trajectory["actions"])
            self.terminated.append(trajectory["terminated"])
            self.truncated.append(trajectory["truncated"])

            # handle data that might optionally be in the trajectory
            if "rewards" in trajectory:
                if self.rewards is None:
                    self.rewards = [trajectory["rewards"]]
                else:
                    self.rewards.append(trajectory["rewards"])
            if "success" in trajectory:
                if self.success is None:
                    self.success = [trajectory["success"]]
                else:
                    self.success.append(trajectory["success"])
            if "fail" in trajectory:
                if self.fail is None:
                    self.fail = [trajectory["fail"]]
                else:
                    self.fail.append(trajectory["fail"])
        def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x
        # uint16 dtype is used to conserve disk space and memory
        # you can optimize this dataset code to keep it as uint16 and process that
        # dtype of data yourself. for simplicity we simply cast to a int32 so
        # it can automatically be converted to torch tensors without complaint
        for i in range(len(self.obs)):
            self.obs[i] = remove_np_uint16(self.obs[i])
                

    def get_sequence_dataset(self):
        import copy
        new_dataset = copy.copy(self)
        sequence_obs = None
        for obs in new_dataset.obs:
            if sequence_obs is None:
                sequence_obs = obs
            else:
                sequence_obs = common.append_dict_array(sequence_obs, obs)
        new_dataset.obs = sequence_obs
        new_dataset.actions = np.vstack(new_dataset.actions)
        new_dataset.terminated = np.concatenate(new_dataset.terminated)
        new_dataset.truncated = np.concatenate(new_dataset.truncated)
        if new_dataset.rewards is not None:
            new_dataset.rewards = np.concatenate(new_dataset.rewards)
        if new_dataset.success is not None:
            new_dataset.success = np.concatenate(new_dataset.success)
        if new_dataset.fail is not None:
            new_dataset.fail = np.concatenate(new_dataset.fail)
        new_dataset.is_episode_dataset = False
        return new_dataset
      
        
    def load_dataset(self, load_count, success_only, device):
        self.obs = None
        self.actions = []
        self.terminated = []
        self.truncated = []
        self.success, self.fail, self.rewards = None, None, None
        if load_count == -1:
            load_count = len(self.episodes)
        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            if success_only: 
                assert "success" in eps, "episodes in this dataset do not have the success attribute, cannot load dataset with success_only=True"
                if not eps["success"]:
                    continue
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)
            eps_len = len(trajectory["actions"])
            
            # exclude the final observation as most learning workflows do not use it
            obs = common.index_dict_array(trajectory["obs"], slice(eps_len))
            if eps_id == 0:
                self.obs = obs
            else:
                self.obs = common.append_dict_array(self.obs, obs)

            self.actions.append(trajectory["actions"])
            self.terminated.append(trajectory["terminated"])
            self.truncated.append(trajectory["truncated"])

            # handle data that might optionally be in the trajectory
            if "rewards" in trajectory:
                if self.rewards is None:
                    self.rewards = [trajectory["rewards"]]
                else:
                    self.rewards.append(trajectory["rewards"])
            if "success" in trajectory:
                if self.success is None:
                    self.success = [trajectory["success"]]
                else:
                    self.success.append(trajectory["success"])
            if "fail" in trajectory:
                if self.fail is None:
                    self.fail = [trajectory["fail"]]
                else:
                    self.fail.append(trajectory["fail"])

        self.actions = np.vstack(self.actions)
        self.terminated = np.concatenate(self.terminated)
        self.truncated = np.concatenate(self.truncated)
        
        if self.rewards is not None:
            self.rewards = np.concatenate(self.rewards)
        if self.success is not None:
            self.success = np.concatenate(self.success)
        if self.fail is not None:
            self.fail = np.concatenate(self.fail)

        def remove_np_uint16(x: Union[np.ndarray, dict]):
            if isinstance(x, dict):
                for k in x.keys():
                    x[k] = remove_np_uint16(x[k])
                return x
            else:
                if x.dtype == np.uint16:
                    return x.astype(np.int32)
                return x
        
        # uint16 dtype is used to conserve disk space and memory
        # you can optimize this dataset code to keep it as uint16 and process that
        # dtype of data yourself. for simplicity we simply cast to a int32 so
        # it can automatically be converted to torch tensors without complaint
        self.obs = remove_np_uint16(self.obs)

        if device is not None:
            self.actions = to_tensors(self.actions, device=device)
            self.obs = to_tensors(self.obs, device=device)
            self.terminated = to_tensors(self.terminated, device=device)
            self.truncated = to_tensors(self.truncated, device=device)
            if self.rewards is not None:
                self.rewards = to_tensors(self.rewards, device=device)
            if self.success is not None:
                self.success = to_tensors(self.terminated, device=device)
            if self.fail is not None:
                self.fail = to_tensors(self.truncated, device=device)
                
                
    def __len__(self):
        return len(self.actions)


    def __getitem__(self, idx):
        if self.is_episode_dataset:
            return self.get_episode(idx)
        else:
            return self.get_step(idx)


    def get_episode(self, idx):
        return self.obs[idx], self.actions[idx], self.terminated[idx], self.truncated[idx]


    def get_step(self, idx):
        action = to_tensors(self.actions[idx], device=self.device)
        obs = common.index_dict_array(self.obs, idx, inplace=False)

        res = dict(
            obs=obs,
            action=action,
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
        )
        if self.rewards is not None:
            res.update(reward=self.rewards[idx])
        if self.success is not None:
            res.update(success=self.success[idx])
        if self.fail is not None:
            res.update(fail=self.fail[idx])
        return res
    

def quat_to_rpy(quaternion, degrees: bool = True):
    rotation_object = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    rpy = rotation_object.as_euler('xyz', degrees=degrees)
    return rpy

INSTRUCTIONS = {
    "StackCube-v1": "stack the red cube on top of the green one",
    "PickCube-v1": "pick up the red cube",
    'PushCube-v1': 'push the cube to the target position',
}

class InternVLPretrainDatasetGenerator:
    def __init__(self, dataset: ManiSkillTrajectoryDataset, save_path: str, horizon=1, dual_camera=False, is_joint_action=False, env_id="StackCube-v1"):
        """
        Initializes the dataset generator for InternVL pretraining.

        Args:
            dataset_path (str): Path to the dataset file.
            load_count (int): Number of trajectories to load. If -1, loads all.
            success_only (bool): Whether to filter for successful trajectories only.
            device: Device to load data onto (e.g., 'cpu', 'cuda').
        """
        self.dataset = dataset
        self.save_path = save_path
        self.img_save_path = os.path.join(save_path, "images")
        os.makedirs(self.img_save_path, exist_ok=True)
        rng = np.random.default_rng(0)
        num_steps = np.sum([len(act) for act in dataset.actions])
        self.val_ids = set(rng.choice(num_steps, size=num_steps // 20, replace=False))
        self.rng = rng
        self.rescale_array = {
            'action': np.array([1000] * 7 + [1]) if is_joint_action else np.array([1000, 1000, 1000, 57.3, 57.3, 57.3] + [1]),
            # 'action': np.array([1000] * 7 + [1]) if is_joint_action else np.array([1000, 1000, 1000, 1000, 1000, 1000] + [1]),
            'qpos': np.array([1000] * 9),
            'tcp_pose': np.array([1000] * 7)
        }
        self.horizon = horizon
        self.dual_camera = dual_camera
        self.instruction = INSTRUCTIONS[env_id]
        
    def cal_statistics(self):
        statistics = {
            'action': {},
            'qpos': {},
            'tcp_pose': {}
        }
        all_collections = {
            'action': [],
            'qpos': [],
            'tcp_pose': []
        }
        for data in self.dataset:
            # camera = data['sensor_data']["base_camera"]["rgb"]
            qpos = data['obs']["agent"]["qpos"]
            tcp_pose = data['obs']["extra"]["tcp_pose"]
            rpy = quat_to_rpy(tcp_pose[3:7], degrees=True)
            tcp_pose = np.concatenate([tcp_pose[:3], rpy])
            action = data["action"]
            all_collections['action'].append(action)
            all_collections['qpos'].append(qpos)
            all_collections['tcp_pose'].append(tcp_pose)

        for key in statistics.keys():
            all_data = all_collections[key]
            statistics[key]['mean'] = np.mean(all_data, axis=0).tolist()
            statistics[key]['std'] = np.std(all_data, axis=0).tolist()
            statistics[key]['min'] = np.min(all_data, axis=0).tolist()
            statistics[key]['max'] = np.max(all_data, axis=0).tolist()
        # print("Statistics calculated:", statistics)
        self.statistics = statistics
        with open(os.path.join(self.save_path, 'statistics.json'), 'w') as f:
            json.dump(statistics, f, indent=4)
     
    def rescale(self, data, key):
        _tmp_data = np.round(data * self.rescale_array[key]).astype(np.int32)
        if key == "action":
            _tmp_data = np.clip(-999, 999, _tmp_data)
        return _tmp_data
    
    def process_episode_data(self, episode_data, filter_zero=False):
        infos = []
        for i in range(len(episode_data['queries'])):
            if self.horizon > 1:
                if i + self.horizon > len(episode_data['queries']):
                    responses = " ".join(episode_data['action_strs'][i:])
                    delta = i + self.horizon - len(episode_data['queries'])
                    responses += f" +0 +0 +0 +0 +0 +0 {episode_data['action_strs'][-1][-3:-1]}|" * delta
                else:
                    responses = " ".join(episode_data['action_strs'][i:i+self.horizon])
            else:
                responses = episode_data['action_strs'][i]
                if filter_zero:
                    if responses[:12] == " 0 0 0 0 0 0":
                        continue
            data_dict = {
                'query': episode_data['queries'][i],
                'response': responses,
                'images': episode_data['cameras_save_path'][i]
            }
            infos.append(data_dict)
        return infos

    def traj_generation(self, filter_zero=False):
        all_infos = []
        val_infos = []
        train_infos = []
        for episode_idx, (obs, action, terminated, truncated) in enumerate(self.dataset):
            episode_data = {
                "queries": [],
                "action_strs": [],
                "cameras_save_path": [],
            }
            cameras = obs['sensor_data']["base_camera"]["rgb"]
            if "hand_camera" in obs['sensor_data']:
                hand_cameras = obs['sensor_data']["hand_camera"]["rgb"]
            else:
                hand_cameras = [None] * len(cameras)
            qposes = obs["agent"]["qpos"]
            tcp_pose = obs["extra"]["tcp_pose"]
            rescaled_qposes = self.rescale(qposes, 'qpos')
            # rescaled_tcp_poses = self.rescale(tcp_poses, 'tcp_pose')
            rescaled_actions = self.rescale(action, 'action')
            rpys = []
            for quat in tcp_pose[:, 3:7]:
                rpy = quat_to_rpy(quat, degrees=True)
                rpys.append(rpy)
            rpys = np.array(rpys)
            tcp_pose = np.concatenate([tcp_pose[:, :3], rpys], axis=-1)
            tcp_pose[:, :3] = np.round(tcp_pose[:, :3] * 1000).astype(np.int32)
            tcp_pose[:, 3:] = np.round(tcp_pose[:, 3:]).astype(np.int32)
            
            for local_step, (camera, hand_camera, rescaled_action, rescaled_qpos, rescaled_tcp_pose) in enumerate(zip(cameras, hand_cameras, rescaled_actions, rescaled_qposes, tcp_pose)): 
                if np.all(rescaled_action[:6] == 0):
                    continue
                camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_0.jpg")
                hand_camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_1.jpg")
                camera_save_name = camera_save_path
                hand_camera_save_name = hand_camera_save_path
                if not os.path.exists(camera_save_path):
                    img = Image.fromarray(camera)
                    img.save(camera_save_path)
                if not os.path.exists(hand_camera_save_path) and hand_camera is not None:
                    hand_img = Image.fromarray(hand_camera)
                    hand_img.save(hand_camera_save_path)
                if rescaled_qpos[-1] >= 37:
                    gripper_state = 1
                else:
                    gripper_state = 0
                eef_xyz = rescaled_tcp_pose[:3]
                eef_rpy = rescaled_tcp_pose[3:]
                # query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                # query = f"The current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm, roll: {eef_rpy[0]} degrees, pitch: {eef_rpy[1]} degrees, yaw: {eef_rpy[2]} degrees, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                # joints_str = ", ".join(f"Joint_{i}: {v}" for i, v in enumerate(rescaled_qpos[:8]))
                # query = f"The current position state of the robotic arm's end gripper is as follows: {{{joints_str}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                # action_str = " ".join([f"{str(int(a))}" for a in rescaled_action])
                rescaled_action[-1] = 0 if rescaled_action[-1] == -1 else 1
                query = f"The current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm, roll: {eef_rpy[0]} degrees, pitch: {eef_rpy[1]} degrees, yaw: {eef_rpy[2]} degrees, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                action_str = f"action: {{x: {rescaled_action[0]}mm, y: {rescaled_action[1]}mm, z: {rescaled_action[2]}mm, roll: {rescaled_action[3]} degrees, pitch: {rescaled_action[4]} degrees, yaw: {rescaled_action[5]} degrees, open: {rescaled_action[6]}}}"
                action_str = action_str
                # if rescaled_action[0] >= 0:
                #     action_str = " " + action_str
                episode_data['queries'].append(query)
                episode_data['action_strs'].append(action_str)
                if self.dual_camera:
                    camera_save_names = [camera_save_name, hand_camera_save_name]
                else: 
                    camera_save_names = [camera_save_name]
                episode_data['cameras_save_path'].append(camera_save_names)
            infos = self.process_episode_data(episode_data, filter_zero)
            all_infos.extend(infos)
        if self.dual_camera:
            prefix = "dualcam_"
        else:
            prefix = ""
        for idx in range(len(all_infos)):
            if idx in self.val_ids:
                val_infos.append(all_infos[idx])
            else:
                train_infos.append(all_infos[idx])
        with open(os.path.join(self.save_path, prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(train_infos, f, indent=4)
        with open(os.path.join(self.save_path, "val_" + prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(val_infos, f, indent=4)
        
    
    def generation(self):
        json_infos = []
        val_infos = []
        for idx, data in enumerate(self.dataset):
            data_dict = {}
            is_done = data["terminated"] or data["truncated"]
            camera = data["obs"]['sensor_data']["base_camera"]["rgb"]
            hand_camera = data["obs"]['sensor_data']["hand_camera"]["rgb"]
            qpos = data["obs"]["agent"]["qpos"]
            tcp_pose = data["obs"]["extra"]["tcp_pose"]
            action = data["action"]
            rescaled_qpos = self.rescale(qpos, 'qpos')
            rescaled_tcp_pose = self.rescale(tcp_pose, 'tcp_pose')
            rescaled_action = self.rescale(action, 'action')
            if not os.path.exists(os.path.join(self.img_save_path, f"{idx}.jpg")):
                img = Image.fromarray(camera)
                img.save(os.path.join(self.img_save_path, f"{idx}.jpg"))
            if not os.path.exists(os.path.join(self.img_save_path, f"{idx}_hand.jpg")):
                hand_img = Image.fromarray(hand_camera)
                hand_img.save(os.path.join(self.img_save_path, f"{idx}_hand.jpg"))
            
            # query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            # query = f"The current position state of the robotic arm's end gripper is as follows: {{Joint_0: {rescaled_qpos[0]}, Joint_1: {rescaled_qpos[1]}, Joint_2: {rescaled_qpos[2]}, Joint_3: {rescaled_qpos[3]}, Joint_4: {rescaled_qpos[4]}, Joint_5: {rescaled_qpos[5]}, Joint_6: {rescaled_qpos[6]}, Joint_7: {rescaled_qpos[7]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            query = f"Based on whether the robot arm's gripper successfully grasps the object and the distance between the robot arm's endpoint and the target position in the image, provide a comprehensive rating, where a higher score indicates better task completion (max score 900). Please rate the completion of the robot arm with gripper follows instruction: sweep the trash to the white bin.\nThe current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm, roll: {eef_rpy[0]} degrees, pitch: {eef_rpy[1]} degrees, yaw: {eef_rpy[2]} degrees, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            action_str = f"action: {{x: {rescaled_action[0]}mm, y: {rescaled_action[1]}mm, z: {rescaled_action[2]}mm, roll: {rescaled_action[3]} degrees, pitch: {rescaled_action[4]} degrees, yaw: {rescaled_action[5]} degrees, open: {rescaled_action[6]}}}"
            # action_str = " ".join([f"{str(int(a))}" for a in rescaled_action])
            # action_str = '{' + action_str + '}'
            data_dict['query'] = query
            data_dict['response'] = action_str
            data_dict['images'] = [os.path.join(self.img_save_path, f"{idx}.jpg"), os.path.join(self.img_save_path, f"{idx}_hand.jpg")]
            if idx in self.val_ids:
                val_infos.append(data_dict)
            else:
                json_infos.append(data_dict)
        if self.dual_camera:
            prefix = "dualcam_"
        else:
            prefix = ""
        with open(os.path.join(self.save_path, prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(json_infos, f, indent=4)
        with open(os.path.join(self.save_path, "val_" + prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(val_infos, f, indent=4)
            

class InternVLPretrainDatasetGeneratorNew:
    def __init__(self, dataset: ManiSkillTrajectoryDataset, save_path: str, horizon=1, dual_camera=False, is_joint_action=False, env_id="StackCube-v1"):
        """
        Initializes the dataset generator for InternVL pretraining.

        Args:
            dataset_path (str): Path to the dataset file.
            load_count (int): Number of trajectories to load. If -1, loads all.
            success_only (bool): Whether to filter for successful trajectories only.
            device: Device to load data onto (e.g., 'cpu', 'cuda').
        """
        self.dataset = dataset
        self.save_path = save_path
        self.img_save_path = os.path.join(save_path, "images")
        os.makedirs(self.img_save_path, exist_ok=True)
        rng = np.random.default_rng(0)
        num_steps = np.sum([len(act) for act in dataset.actions])
        self.val_ids = set(rng.choice(num_steps, size=num_steps // 20, replace=False))
        self.rng = rng
        self.rescale_array = {
            'action': np.array([1000] * 7 + [1]) if is_joint_action else np.array([1000, 1000, 1000, 57.3, 57.3, 57.3] + [1]),
            # 'action': np.array([1000] * 7 + [1]) if is_joint_action else np.array([1000, 1000, 1000, 1000, 1000, 1000] + [1]),
            'qpos': np.array([1000] * 9),
            'tcp_pose': np.array([1000] * 7)
        }
        self.horizon = horizon
        self.dual_camera = dual_camera
        self.instruction = INSTRUCTIONS[env_id]
        
    def cal_statistics(self):
        statistics = {
            'action': {},
            'qpos': {},
            'tcp_pose': {}
        }
        all_collections = {
            'action': [],
            'qpos': [],
            'tcp_pose': []
        }
        for data in self.dataset:
            # camera = data['sensor_data']["base_camera"]["rgb"]
            qpos = data['obs']["agent"]["qpos"]
            tcp_pose = data['obs']["extra"]["tcp_pose"]
            rpy = quat_to_rpy(tcp_pose[3:7], degrees=True)
            tcp_pose = np.concatenate([tcp_pose[:3], rpy])
            action = data["action"]
            all_collections['action'].append(action)
            all_collections['qpos'].append(qpos)
            all_collections['tcp_pose'].append(tcp_pose)

        for key in statistics.keys():
            all_data = all_collections[key]
            statistics[key]['mean'] = np.mean(all_data, axis=0).tolist()
            statistics[key]['std'] = np.std(all_data, axis=0).tolist()
            statistics[key]['min'] = np.min(all_data, axis=0).tolist()
            statistics[key]['max'] = np.max(all_data, axis=0).tolist()
        # print("Statistics calculated:", statistics)
        self.statistics = statistics
        with open(os.path.join(self.save_path, 'statistics.json'), 'w') as f:
            json.dump(statistics, f, indent=4)
     
    def rescale(self, data, key):
        _tmp_data = np.round(data * self.rescale_array[key]).astype(np.int32)
        if key == "action":
            _tmp_data = np.clip(-999, 999, _tmp_data)
        return _tmp_data
    
    def process_episode_data(self, episode_data, filter_zero=False):
        infos = []
        for i in range(len(episode_data['queries'])):
            if self.horizon > 1:
                if i + self.horizon > len(episode_data['queries']):
                    responses = " ".join(episode_data['action_strs'][i:])
                    delta = i + self.horizon - len(episode_data['queries'])
                    responses += f" +0 +0 +0 +0 +0 +0 {episode_data['action_strs'][-1][-3:-1]}|" * delta
                else:
                    responses = " ".join(episode_data['action_strs'][i:i+self.horizon])
            else:
                responses = episode_data['action_strs'][i]
                if filter_zero:
                    if responses[:12] == " 0 0 0 0 0 0":
                        continue
            data_dict = {
                'query': episode_data['queries'][i],
                'response': responses,
                'images': episode_data['cameras_save_path'][i]
            }
            infos.append(data_dict)
        return infos

    def traj_generation(self, filter_zero=False):
        all_infos = []
        val_infos = []
        train_infos = []
        for episode_idx, (obs, action, terminated, truncated) in enumerate(self.dataset):
            episode_data = {
                "queries": [],
                "action_strs": [],
                "cameras_save_path": [],
            }
            cameras = obs['sensor_data']["base_camera"]["rgb"]
            if "hand_camera" in obs['sensor_data']:
                hand_cameras = obs['sensor_data']["hand_camera"]["rgb"]
            else:
                hand_cameras = [None] * len(cameras)
            qposes = obs["agent"]["qpos"]
            tcp_pose = obs["extra"]["tcp_pose"]
            rescaled_qposes = self.rescale(qposes, 'qpos')
            # rescaled_tcp_poses = self.rescale(tcp_poses, 'tcp_pose')
            rescaled_actions = self.rescale(action, 'action')
            rpys = []
            for quat in tcp_pose[:, 3:7]:
                rpy = quat_to_rpy(quat, degrees=True)
                rpys.append(rpy)
            rpys = np.array(rpys)
            tcp_pose = np.concatenate([tcp_pose[:, :3], rpys], axis=-1)
            tcp_pose[:, :3] = np.round(tcp_pose[:, :3] * 1000).astype(np.int32)
            tcp_pose[:, 3:] = np.round(tcp_pose[:, 3:]).astype(np.int32)
            len_episode = len(cameras)
            for local_step, (camera, hand_camera, rescaled_action, rescaled_qpos, rescaled_tcp_pose) in enumerate(zip(cameras, hand_cameras, rescaled_actions, rescaled_qposes, tcp_pose)): 
                if np.all(rescaled_action[:6] == 0):
                    continue
                camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_0.jpg")
                hand_camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_1.jpg")
                camera_save_name = camera_save_path
                hand_camera_save_name = hand_camera_save_path
                if not os.path.exists(camera_save_path):
                    img = Image.fromarray(camera)
                    img.save(camera_save_path)
                if not os.path.exists(hand_camera_save_path) and hand_camera is not None:
                    hand_img = Image.fromarray(hand_camera)
                    hand_img.save(hand_camera_save_path)
                if rescaled_qpos[-1] >= 37:
                    gripper_state = 1
                else:
                    gripper_state = 0
                eef_xyz = rescaled_tcp_pose[:3]
                eef_rpy = rescaled_tcp_pose[3:]
                # query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                query = f"Based on whether the robot arm's gripper successfully grasps the object and the distance between the robot arm's endpoint and the target position in the image, provide a comprehensive rating, where a higher score indicates better task completion (max score 900). Please rate the completion of the robot arm with gripper follows instruction: sweep the trash to the white bin.\nThe current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm, roll: {eef_rpy[0]} degrees, pitch: {eef_rpy[1]} degrees, yaw: {eef_rpy[2]} degrees, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                # joints_str = ", ".join(f"Joint_{i}: {v}" for i, v in enumerate(rescaled_qpos[:8]))
                # query = f"The current position state of the robotic arm's end gripper is as follows: {{{joints_str}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                
                # action_str = " ".join([f"{str(int(a))}" for a in rescaled_action])
                # action_str = action_str
                rescaled_action[-1] = 0 if rescaled_action[-1] == -1 else 1
                action_str = f"completion value:{901 - len_episode + local_step}\naction: {{x: {rescaled_action[0]}mm, y: {rescaled_action[1]}mm, z: {rescaled_action[2]}mm, roll: {rescaled_action[3]} degrees, pitch: {rescaled_action[4]} degrees, yaw: {rescaled_action[5]} degrees, open: {rescaled_action[6]}}}"
                # if rescaled_action[0] >= 0:
                #     action_str = " " + action_str
                episode_data['queries'].append(query)
                episode_data['action_strs'].append(action_str)
                if self.dual_camera:
                    camera_save_names = [camera_save_name, hand_camera_save_name]
                else: 
                    camera_save_names = [camera_save_name]
                episode_data['cameras_save_path'].append(camera_save_names)
            infos = self.process_episode_data(episode_data, filter_zero)
            all_infos.extend(infos)
        if self.dual_camera:
            prefix = "dualcam_"
        else:
            prefix = ""
        for idx in range(len(all_infos)):
            if idx in self.val_ids:
                val_infos.append(all_infos[idx])
            else:
                train_infos.append(all_infos[idx])
        with open(os.path.join(self.save_path, prefix + f'new_dataset_{self.horizon}.json'), 'w') as f:
            json.dump(train_infos, f, indent=4)
        with open(os.path.join(self.save_path, "val_" + prefix + f'new_dataset_{self.horizon}.json'), 'w') as f:
            json.dump(val_infos, f, indent=4)
        
    
    def generation(self):
        json_infos = []
        val_infos = []
        for idx, data in enumerate(self.dataset):
            data_dict = {}
            is_done = data["terminated"] or data["truncated"]
            camera = data["obs"]['sensor_data']["base_camera"]["rgb"]
            hand_camera = data["obs"]['sensor_data']["hand_camera"]["rgb"]
            qpos = data["obs"]["agent"]["qpos"]
            tcp_pose = data["obs"]["extra"]["tcp_pose"]
            action = data["action"]
            rescaled_qpos = self.rescale(qpos, 'qpos')
            rescaled_tcp_pose = self.rescale(tcp_pose, 'tcp_pose')
            rescaled_action = self.rescale(action, 'action')
            if not os.path.exists(os.path.join(self.img_save_path, f"{idx}.jpg")):
                img = Image.fromarray(camera)
                img.save(os.path.join(self.img_save_path, f"{idx}.jpg"))
            if not os.path.exists(os.path.join(self.img_save_path, f"{idx}_hand.jpg")):
                hand_img = Image.fromarray(hand_camera)
                hand_img.save(os.path.join(self.img_save_path, f"{idx}_hand.jpg"))
            
            # query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            query = f"The current position state of the robotic arm's end gripper is as follows: {{Joint_0: {rescaled_qpos[0]}, Joint_1: {rescaled_qpos[1]}, Joint_2: {rescaled_qpos[2]}, Joint_3: {rescaled_qpos[3]}, Joint_4: {rescaled_qpos[4]}, Joint_5: {rescaled_qpos[5]}, Joint_6: {rescaled_qpos[6]}, Joint_7: {rescaled_qpos[7]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"

            action_str = " ".join([f"{str(int(a))}" for a in rescaled_action])
            action_str = '{' + action_str + '}'
            data_dict['query'] = query
            data_dict['response'] = action_str
            data_dict['images'] = [os.path.join(self.img_save_path, f"{idx}.jpg"), os.path.join(self.img_save_path, f"{idx}_hand.jpg")]
            if idx in self.val_ids:
                val_infos.append(data_dict)
            else:
                json_infos.append(data_dict)
        if self.dual_camera:
            prefix = "dualcam_"
        else:
            prefix = ""
        with open(os.path.join(self.save_path, prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(json_infos, f, indent=4)
        with open(os.path.join(self.save_path, "val_" + prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(val_infos, f, indent=4)
            