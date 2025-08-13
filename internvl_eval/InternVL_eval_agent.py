import torch
from transformers import AutoModel, AutoTokenizer
import os
import json
import numpy as np
import torchvision.transforms as T
import datetime
from collections import defaultdict
import multiprocessing
import gymnasium as gym
import mani_skill.envs
import traceback
import shutil
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from utils import quat_to_rpy, load_image, parse_and_validate_vector, append_to_jsonl, VideoRecorder
from transformers.generation.logits_process import PrefixConstrainedLogitsProcessor, LogitsProcessor
import math
from transformers import LogitsProcessorList

class SafePrefixConstrainedLogitsProcessor(PrefixConstrainedLogitsProcessor):
    def __call__(self, input_ids, scores):
        if input_ids.shape[-1] == 0:
            mask = torch.full_like(scores, -math.inf)
            batch_id = 0
            sent = input_ids[batch_id]
            prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
            mask[..., prefix_allowed_tokens] = 0.0
            return scores + mask
        return super().__call__(input_ids, scores)

def generate_prefix_fn_legacy(numbers_list, start_list, end_list, connect_list):

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        if input_ids.shape[-1] == 14:
            return end_list
        if input_ids.shape[-1] == 0:
            return start_list
        elif input_ids.shape[-1] % 2 == 1:
            return numbers_list
        elif input_ids.shape[-1] % 2 == 0:
            return connect_list
    return prefix_allowed_tokens_fn

def generate_prefix_fn(numbers_list, symbols_list):
    def prefix_allowed_tokens_fn(batch_id, input_ids):
        if input_ids.shape[-1] % 2 == 0:
            return symbols_list
        elif input_ids.shape[-1] % 2 == 1:
            return numbers_list
    return prefix_allowed_tokens_fn


def prepare_logits_processor(is_legacy, tokenizer):
    numbers = list(range(0, 1000))
    processor_list = LogitsProcessorList([])
    if is_legacy:
        print("Using action pattern: {-1 0 0 0 0 0 1}")
        start_list = []
        end_list = []
        connect_list = []
        numbers_list = []
        start_sign = ["{", '{-',]
        end_sign = ["}"]
        connect_sign = [" ", " -"]
        for str_ in start_sign:
            toks = tokenizer.tokenize(str_)
            assert len(toks) == 1
            start_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
        for str_ in end_sign:
            toks = tokenizer.tokenize(str_)
            assert len(toks) == 1
            end_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
        for str_ in connect_sign:
            toks = tokenizer.tokenize(str_)
            assert len(toks) == 1
            connect_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
        for str_ in numbers:
            toks = tokenizer.tokenize(str(str_))
            assert len(toks) == 1
            numbers_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
        prefix_processor = SafePrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=generate_prefix_fn_legacy(numbers_list, start_list, end_list, connect_list),
                num_beams=1,
            )
        processor_list = LogitsProcessorList([
            prefix_processor,
        ])
        valid_list = start_list + end_list + connect_list + numbers_list
    else:
        print("Using action pattern: 0 0 0 0 0 0 1")
        connect_list = []
        numbers_list = []
        connect_sign = [" ", " -"]
        for str_ in connect_sign:
            toks = tokenizer.tokenize(str_)
            assert len(toks) == 1
            connect_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
        for str_ in numbers:
            toks = tokenizer.tokenize(str(str_))
            assert len(toks) == 1
            numbers_list.append(tokenizer.convert_tokens_to_ids(toks)[0])
        prefix_processor = SafePrefixConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=generate_prefix_fn(numbers_list, connect_list),
                num_beams=1,
            )
        processor_list = LogitsProcessorList([
            prefix_processor,
        ])
        valid_list = numbers_list + connect_list
    return processor_list, valid_list

class InternVLEvalAgent:
    def __init__(
        self, 
        model_path,
        instruction=None,
        parent_tag: str = None,
        inference_tag: str = None,
        num_envs: int = 1,
        device: str = "cuda"
    ):
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        modeling_name = "modeling_internvl_chat.py"
        modeling_path = os.path.join(current_file_path, modeling_name)
        modeling_to_replace = os.path.join(model_path, "modeling_internvl_chat.py")
        model_dir_name = model_path.split("/")[1]
        if os.path.islink(modeling_to_replace):
            os.remove(modeling_to_replace)
        if os.path.exists(modeling_to_replace) and not os.path.exists(modeling_to_replace + ".bak"):
            os.rename(modeling_to_replace, modeling_to_replace + ".bak")
            shutil.copy(src=modeling_path, dst=modeling_to_replace)
        elif not os.path.exists(modeling_to_replace):
            shutil.copy(src=modeling_path, dst=modeling_to_replace)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().to(device)
        if "horizon" in model_dir_name:
            self.horizon = int(model_dir_name.split("_")[-1])
        else:
            self.horizon = 1
        if 'joint' in model_dir_name:
            self.joint = True
        else:
            self.joint = False
        if "dual" in model_path:
            self.dual_cam = True
        else:
            self.dual_cam = False
        if self.joint:
            self.action_rescale = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1])
        else:
            # self.action_rescale = np.array([1000, 1000, 1000, 57.3, 57.3, 57.3, 1])
            self.action_rescale = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1])
        self.action_dim = len(self.action_rescale)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        is_legacy_action = True if 'legacy' in model_dir_name else False
        processor_list, valid_list = prepare_logits_processor(is_legacy_action, self.tokenizer)
        self.generation_config = dict(max_new_tokens=15 if is_legacy_action else (self.action_dim * 2 * self.horizon), do_sample=True, logits_processor=processor_list)
        self.instruction = 'stack the red cube on top of the green one' if instruction is None else instruction
        jsonl_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if inference_tag is None else inference_tag
        self.jsonl_path = os.path.join(model_path, parent_tag, jsonl_name, 'inference.jsonl')
        self.num_envs = num_envs
        self.device = device

        self.model_path = model_path



    def get_next_action(self, observations):
        num_patches_list = []
        pixel_values = []
        questions = []
        qposes = []
        for env_id in range(self.num_envs):
            qpos = observations["agent"]["qpos"][env_id].cpu().numpy()
            tcp_pose = observations["extra"]["tcp_pose"][env_id].cpu().numpy()
            qposes.append(qpos)
            camera = observations['sensor_data']["base_camera"]["rgb"][env_id].cpu().numpy()
            eef_xyz = tcp_pose[:3]
            eef_xyz = np.round(eef_xyz * 1000).astype(np.int32)  # Convert to mm
            eef_rpy = quat_to_rpy(tcp_pose[3:7], degrees=True)
            eef_rpy = np.round(eef_rpy).astype(np.int32)  # Convert to degrees
            rescaled_qpos = np.round(qpos * 1000).astype(np.int32)
            if qpos[-1] >= 0.037:
                gripper_state = 1
            else:
                gripper_state = 0
            # if self.joint:
            #     query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            # else:
            #     query = f"The current position state of the robotic arm's end gripper is as follows: {{x: {eef_xyz[0]}mm, y: {eef_xyz[1]}mm, z: {eef_xyz[2]}mm, roll: {eef_rpy[0]} degrees, pitch: {eef_rpy[1]} degrees, yaw: {eef_rpy[2]} degrees, open: {gripper_state}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            # joints_str = ", ".join(f"Joint_{i}: {v}" for i, v in enumerate(rescaled_qpos[:9]))
            joints_str = ", ".join(f"Joint_{i}: {v}" for i, v in enumerate(rescaled_qpos[:8]))
            query = f"The current position state of the robotic arm's end gripper is as follows: {{{joints_str}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
            pixel_0 = load_image(camera, max_num=12).to(torch.bfloat16).to(self.device)
            patch_list = []
            pixels = []
            patch_list.append(pixel_0.size(0))
            pixels.append(pixel_0)
            if self.dual_cam:
                query = "<image><image>" + query
                hand_camera = observations['sensor_data']["hand_camera"]["rgb"][env_id].cpu().numpy()
                pixel_1 = load_image(hand_camera, max_num=12).to(torch.bfloat16).to(self.device)
                patch_list.append(pixel_1.size(0))
                pixels.append(pixel_1)
            else:
                query = "<image>" + query
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
            action_extracted = parse_and_validate_vector(response, self.action_dim)
            if action_extracted is None:
                action_extracted = np.zeros(self.action_dim, dtype=np.float64)
                if qposes[env_id][-1] >= 0.037:
                    action_extracted[-1] = 1
                else:
                    action_extracted[-1] = -1
            else:
                # action_extracted[:-1] = action_extracted[:-1] / 1000
                action_extracted = action_extracted / self.action_rescale
            # print("-----------------------------------------------------")
            action_to_print = [np.round(a, 3) for a in action_extracted.values()] if isinstance(action_extracted, dict) else [np.round(a, 3) for a in action_extracted]
            # print(f'User: {question}\nAction: {action_to_print}')
            # print("-----------------------------------------------------")
            actions.append(action_extracted)
            append_to_jsonl({
                "question": question,
                "response": response,
                "action_vector": [float(a) for a in action_to_print],
                'env_id': env_id,
            }, self.jsonl_path)
        return np.array(actions)
    
def eval_checkpoint(model_parent, ckpt_name, gpu_id, instruction=None):
    # --- Key Change 3: Set the GPU for this specific process ---
    # This MUST be the first thing you do before any CUDA/gym/torch initialization.
    print(f"Process {os.getpid()} starting evaluation of {ckpt_name} on GPU {gpu_id}")
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(gpu_id)
    # Assuming your agent and env setup use the GPU
    # from your_agent_file import InternVLEvalAgent

    inference_tag = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    num_envs = 64
    model_path = os.path.join(model_parent, ckpt_name)
    
    # Wrap the core logic in a try...finally block to ensure cleanup
    parent_tag = "mani_infer"
    max_episode_steps = 100
    eval_steps = 100
    try:
        # It's good practice to pass the device to your agent
        # The agent should then use this device, e.g., 'cuda'
        # Note: After setting CUDA_VISIBLE_DEVICES, GPU 0 for this process *is* the assigned gpu_id
        print("init model:", model_path)
        agent = InternVLEvalAgent(
            model_path=model_path,
            instruction=instruction,
            parent_tag=parent_tag,
            inference_tag=inference_tag,
            num_envs=num_envs,
            device=f"cuda:{gpu_id}",
        )
        print("init model:", model_path, 'done')
        # It's good practice to include the GPU ID in the save path
        save_dir = os.path.join(model_path, parent_tag, f"{inference_tag}")
        video_recorder = VideoRecorder(save_path=os.path.join(save_dir, "videos"), fps=30, num_envs=num_envs)
        
        eval_envs = gym.make(
            "PushCube-v1",
            num_envs=num_envs,
            obs_mode="rgb",
            control_mode="pd_joint_delta_pos" if "joint" in model_path else "pd_ee_delta_pose",
            sensor_configs={'height': 480, 'width': 480},
            max_episode_steps=max_episode_steps,
            reconfiguration_freq=1,
            reward_mode='sparse',
            sim_backend=f'cuda:{gpu_id}',
            render_backend=f'cuda:{gpu_id}',
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
            if 'hand_camera' in last_obs['sensor_data']:
                cameras = [last_obs['sensor_data']["base_camera"]["rgb"].cpu().numpy(), last_obs['sensor_data']["hand_camera"]["rgb"].cpu().numpy()]
            else:
                cameras = [last_obs['sensor_data']["base_camera"]["rgb"].cpu().numpy()]
            video_recorder.append_obs(cameras, reward, is_terminals, actions)
            last_obs = obs
            done_env_ids = np.where(is_terminals)[0]
            for env_id in done_env_ids:
                print(f"GPU {gpu_id}: Env {env_id} terminated, reward: {reward[env_id]}.")
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
    # multiprocessing.set_start_method("spawn", force=True)
    # tasks_to_run = [
    #     ["vlav-project/maniskill_stack_cubes_dual1_4x4/internvl2-2b/v0-20250808-133323", "checkpoint-1600", 6],
    #     ["vlav-project/maniskill_stack_cubes_dual1_legacy1_4x4/internvl2-2b/v0-20250808-141511", "checkpoint-1600", 7],
    # ]
    # print(len(tasks_to_run))
    # with multiprocessing.Pool(processes=len(tasks_to_run)) as pool:
    #     # Use starmap to pass multiple arguments to the worker function
    #     pool.starmap(eval_checkpoint, tasks_to_run)
    # # Define which GPUs to use
    # AVAILABLE_GPUS = [0,1,2,3,4,5,6,7] # Modify this to match your system
    AVAILABLE_GPUS = [0,2,4,5,6,7]
    NUM_GPUS = len(AVAILABLE_GPUS)


    tasks_to_run = []

    model_parents = [
        # "vlav-project/maniskill_joint_dual/internvl2-2b/v0-20250809-021840",
        # "vlav-project/maniskill_joint_epds_dual/internvl2-2b/v0-20250809-030109",
        # "vlav-project/maniskill_legacy_reproduce_dual/internvl2-2b/v0-20250810-005645",
        "vlav-project/train_push_cube500_legacy/internvl2-2b/v0-20250812-011657",
        # "vlav-project/mani_stack_cubes_dual_joint_legacy_reproduce/internvl2-2b/v0-20250810-160449",
        # "vlav-project/mani_new_legacy_reproduce_dual/internvl2-2b/v0-20250810-172603",
    ]
    # instructions = ['stack the red cube on top of the green one'] * len(model_parents)
    instructions = ['push the cube to the target position'] * len(model_parents)
    global_index = 0
    for model_parent, instruction in zip(model_parents, instructions):
        print("Model_path:", model_parent)
        checkpoints = [ckpt for ckpt in os.listdir(model_parent) if ckpt.startswith("checkpoint")]
        checkpoints.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)        
        for i, ckpt_name in enumerate(checkpoints): 
            gpu_id = AVAILABLE_GPUS[global_index % NUM_GPUS] # Cycle through available GPUs
            tasks_to_run.append((model_parent, ckpt_name, gpu_id, instruction))
            global_index += 1
    print(f"Found {len(tasks_to_run)} checkpoints to evaluate on {NUM_GPUS} GPUs.")
    for task in tasks_to_run:
        print(task)
    # 2. Create a process pool and run the tasks in parallel
    with multiprocessing.Pool(processes=NUM_GPUS) as pool:
        # Use starmap to pass multiple arguments to the worker function
        pool.starmap(eval_checkpoint, tasks_to_run[:4])