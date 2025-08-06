from math import e
from operator import is_
from turtle import isdown
from typing import Union
import h5py
from matplotlib.pyplot import step
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from mani_skill.utils.io_utils import load_json
from mani_skill.utils import sapien_utils
from mani_skill.utils import common
from PIL import Image
import os
import json


class InternVLPretrainDatasetGenerator:
    def __init__(self, dataset, save_path: str, horizon=1, dual_camera=False, scale_factor=1000,):
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
            'action': np.array([scale_factor] * 6 + [1]),
            'qpos': np.array([scale_factor] * 9),
            'tcp_pose': np.array([scale_factor] * 7)
        }
        self.horizon = horizon
        self.dual_camera = dual_camera
        self.scale_factor = scale_factor
        if "StackCube-v1" in self.dataset.env_id:
            self.instruction = "stack all the cubes"
        
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
            qpos = data["agent"]["qpos"]
            tcp_pose = data["extra"]["tcp_pose"]
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

        # for key in statistics.keys():
        #     for idx, (max_v, min_v) in enumerate(zip(statistics[key]['max'], statistics[key]['min'])):
        #         if abs(max_v) > 1 or abs(min_v) > 1 and self.rescale_array[key][idx] != 1:
        #             self.rescale_array[key][idx] = 100
     
    def rescale(self, data, key):
        _tmp_data = np.round(data * self.rescale_array[key]).astype(np.int32)
        if key == "action":
            _tmp_data = np.clip(-999, 999, _tmp_data)
        return _tmp_data
    
    def process_episode_data(self, episode_data):
        infos = []
        for i in range(len(episode_data['queries'])):
            if i + self.horizon > len(episode_data['queries']):
                responses = " ".join(episode_data['action_strs'][i:])
                delta = i + self.horizon - len(episode_data['queries'])
                responses += f" +0 +0 +0 +0 +0 +0 {episode_data['action_strs'][-1][-3:-1]}|" * delta
            else:
                responses = " ".join(episode_data['action_strs'][i:i+self.horizon])
            data_dict = {
                'query': episode_data['queries'][i],
                'response': responses,
                'images': episode_data['cameras_save_path'][i]
            }
            infos.append(data_dict)
        return infos

    def traj_generation(self):
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
            hand_cameras = obs['sensor_data']["hand_camera"]["rgb"]
            qposes = obs["agent"]["qpos"]
            tcp_poses = obs["extra"]["tcp_pose"]
            rescaled_qposes = self.rescale(qposes, 'qpos')
            rescaled_tcp_poses = self.rescale(tcp_poses, 'tcp_pose')
            rescaled_actions = self.rescale(action, 'action')
            for local_step, (camera, hand_camera, rescaled_action, rescaled_qpos, rescaled_tcp_pose) in enumerate(zip(cameras, hand_cameras, rescaled_actions, rescaled_qposes, rescaled_tcp_poses)): 
                camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_0.jpg")
                hand_camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_1.jpg")
                camera_save_name = camera_save_path
                hand_camera_save_name = hand_camera_save_path
                if not os.path.exists(camera_save_path):
                    img = Image.fromarray(camera)
                    img.save(camera_save_path)
                if not os.path.exists(hand_camera_save_path):
                    hand_img = Image.fromarray(hand_camera)
                    hand_img.save(hand_camera_save_path)
                query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                action_str = " ".join([f"+{str(int(a))}" if a >= 0 else f"{str(int(a))}" for a in rescaled_action])
                action_str = action_str + '|'
                episode_data['queries'].append(query)
                episode_data['action_strs'].append(action_str)
                episode_data['cameras_save_path'].append([camera_save_name, hand_camera_save_name])
            infos = self.process_episode_data(episode_data)
            all_infos.extend(infos)
        if self.dual_camera:
            prefix = "dualcam_"
        else:
            prefix = ""
        if self.scale_factor == 100:
            prefix += "100"
        else:
            prefix += "1000"
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
            
            query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
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
        if self.scale_factor == 100:
            prefix += "100"
        else:
            prefix += "1000"
        with open(os.path.join(self.save_path, prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(json_infos, f, indent=4)
        with open(os.path.join(self.save_path, "val_" + prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(val_infos, f, indent=4)
            
                      

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

    def __init__(self, dataset_file: str, load_count=-1, success_only: bool = False, device = None, is_episode_dataset=False) -> None:
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

class InternVLPretrainDatasetGenerator:
    def __init__(self, dataset: ManiSkillTrajectoryDataset, save_path: str, horizon=1, dual_camera=False, scale_factor=1000,):
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
        try:
            num_steps = np.sum([len(act) for act in dataset.actions])
        except:
            num_steps = len(dataset.actions)
        self.val_ids = set(rng.choice(num_steps, size=num_steps // 20, replace=False))
        self.rng = rng
        self.rescale_array = {
            'action': np.array([scale_factor] * 6 + [1]),
            'qpos': np.array([scale_factor] * 9),
            'tcp_pose': np.array([scale_factor] * 7)
        }
        self.horizon = horizon
        self.dual_camera = dual_camera
        self.scale_factor = scale_factor
        if "StackCube-v1" in self.dataset.env_id:
            self.instruction = "stack all the cubes"
        
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
            qpos = data["agent"]["qpos"]
            tcp_pose = data["extra"]["tcp_pose"]
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

        # for key in statistics.keys():
        #     for idx, (max_v, min_v) in enumerate(zip(statistics[key]['max'], statistics[key]['min'])):
        #         if abs(max_v) > 1 or abs(min_v) > 1 and self.rescale_array[key][idx] != 1:
        #             self.rescale_array[key][idx] = 100
     
    def rescale(self, data, key):
        _tmp_data = np.round(data * self.rescale_array[key]).astype(np.int32)
        if key == "action":
            _tmp_data = np.clip(-999, 999, _tmp_data)
        return _tmp_data
    
    def process_episode_data(self, episode_data):
        infos = []
        for i in range(len(episode_data['queries'])):
            if i + self.horizon > len(episode_data['queries']):
                responses = " ".join(episode_data['action_strs'][i:])
                delta = i + self.horizon - len(episode_data['queries'])
                responses += f" +0 +0 +0 +0 +0 +0 {episode_data['action_strs'][-1][-3:-1]}|" * delta
            else:
                responses = " ".join(episode_data['action_strs'][i:i+self.horizon])
            data_dict = {
                'query': episode_data['queries'][i],
                'response': responses,
                'images': episode_data['cameras_save_path'][i]
            }
            infos.append(data_dict)
        return infos

    def traj_generation(self):
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
            hand_cameras = obs['sensor_data']["hand_camera"]["rgb"]
            qposes = obs["agent"]["qpos"]
            tcp_poses = obs["extra"]["tcp_pose"]
            rescaled_qposes = self.rescale(qposes, 'qpos')
            rescaled_tcp_poses = self.rescale(tcp_poses, 'tcp_pose')
            rescaled_actions = self.rescale(action, 'action')
            for local_step, (camera, hand_camera, rescaled_action, rescaled_qpos, rescaled_tcp_pose) in enumerate(zip(cameras, hand_cameras, rescaled_actions, rescaled_qposes, rescaled_tcp_poses)): 
                camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_0.jpg")
                hand_camera_save_path = os.path.join(self.img_save_path, f"{episode_idx}_{local_step}_1.jpg")
                camera_save_name = camera_save_path
                hand_camera_save_name = hand_camera_save_path
                if not os.path.exists(camera_save_path):
                    img = Image.fromarray(camera)
                    img.save(camera_save_path)
                if not os.path.exists(hand_camera_save_path):
                    hand_img = Image.fromarray(hand_camera)
                    hand_img.save(hand_camera_save_path)
                query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
                action_str = " ".join([f"+{str(int(a))}" if a >= 0 else f"{str(int(a))}" for a in rescaled_action])
                action_str = action_str + '|'
                episode_data['queries'].append(query)
                episode_data['action_strs'].append(action_str)
                episode_data['cameras_save_path'].append([camera_save_name, hand_camera_save_name])
            infos = self.process_episode_data(episode_data)
            all_infos.extend(infos)
        if self.dual_camera:
            prefix = "dualcam_"
        else:
            prefix = ""
        if self.scale_factor == 100:
            prefix += "100"
        else:
            prefix += "1000"
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
            
            query = f"The current joint state of the robotic arm is as follows: {{{rescaled_qpos[0]} {rescaled_qpos[1]} {rescaled_qpos[2]} {rescaled_qpos[3]} {rescaled_qpos[4]} {rescaled_qpos[5]} {rescaled_qpos[6]} {rescaled_qpos[7]} {rescaled_qpos[8]}}}. What action should the robot take to get better completion of instruction: {self.instruction}?"
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
        if self.scale_factor == 100:
            prefix += "100"
        with open(os.path.join(self.save_path, prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(json_infos, f, indent=4)
        with open(os.path.join(self.save_path, "val_" + prefix + f'dataset_{self.horizon}.json'), 'w') as f:
            json.dump(val_infos, f, indent=4)
            
                      