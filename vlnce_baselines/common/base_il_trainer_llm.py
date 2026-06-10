import json
import sys
import jsonlines
import os
import time
import warnings

# Resolve project root for shared model paths (cross-platform)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
from collections import defaultdict
from typing import Dict, List
from PIL import Image
import requests
from openai import OpenAI
from scipy.spatial import cKDTree
import random
import cv2
import open3d as o3d


# for navigator      
from vlnce_baselines.common.navigator.spatialNavigator import *
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as distr
import torch.multiprocessing as mp
import gzip
import math
from copy import deepcopy

import tqdm
from gym import Space
from habitat import Config, logger
from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.base_il_trainer import BaseILTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_extensions.measures import Position
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs, generate_video
from habitat_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)

from habitat_extensions.utils import observations_to_image, colorize_draw_agent_and_fit_to_height
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.env_utils import (
    construct_envs_auto_reset_false,
    construct_envs,
    is_slurm_batch_job,
)
from vlnce_baselines.common.utils import *
from vlnce_baselines.common.map import get_structure_wp

from habitat_extensions.measures import NDTW
from fastdtw import fastdtw

from ..utils import get_camera_orientations
from ..models.utils import (
    length2mask, dir_angle_feature, dir_angle_feature_with_ele,
)

# with warnings.catch_warnings():
#     warnings.filterwarnings("ignore", category=FutureWarning)
#     import tensorflow as tf  # noqa: F401

class BaseVLNCETrainerLLM(BaseILTrainer):
    r"""A base trainer for VLN-CE imitation learning."""
    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.policy = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.obs_transforms = []
        self.start_epoch = 0
        self.step_id = 0

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ) -> None:
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from waypoint_prediction.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        self.waypoint_predictor.load_state_dict(
            torch.load(
                os.path.join(PROJECT_ROOT, "models", "waypoint_prediction", "checkpoints", "check_val_best_avg_wayscore"),
                map_location = torch.device('cpu'),
            )['predictor']['state_dict']
        )
        for param in self.waypoint_predictor.parameters():
            param.requires_grad = False

  
        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)
        self.num_recurrent_layers = self.policy.net.num_recurrent_layers

        logger.info("Finished setting up waypoint_predictor.")

    def load_checkpoint(self, checkpoint_path, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    @staticmethod
    def _pause_envs(
        envs_to_pause,
        envs,
        not_done_masks,
        prev_actions,
        batch,
        rgb_frames=None,
    ):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
                
            not_done_masks = not_done_masks[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            if rgb_frames is not None:
                rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            not_done_masks,
            prev_actions,
            batch,
            rgb_frames,
        )
        
    def generate_input(self, observations):
        instruction = observations['instruction']['text']
        image_dict = {} 
        rgb_image_dict = {}
        depth_image_dict = {}
        rgb_index = 0
        depth_index = 0
        for key in observations.keys():
            image_path = "./image_show/"
            if 'rgb' in key:
                image_path += f"{key}.jpg"
                image = Image.fromarray(observations[key], mode="RGB")
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path, format="JPEG")
                rgb_image_dict[str(rgb_index)] = Image.open(image_path)
                rgb_index += 1
            if 'depth' in key:
                image_path += f"{key}.jpg"
                if observations[key].ndim == 3 and observations[key].shape[-1] == 1:
                    depth_map = observations[key].squeeze(-1)
                depth_img = (255 * (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))).astype(np.uint8)
                image = Image.fromarray(depth_img)
                dir_name = os.path.dirname(image_path)
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                image.save(image_path)
                depth_image_dict[str(depth_index)] = Image.open(image_path)
                depth_index += 1
        for index in rgb_image_dict:
            image_dict[index] = {
                'rgb': rgb_image_dict[index],
                'depth': depth_image_dict[index]
            }
            
        return instruction, image_dict
    

    def construct_image_dicts(self, batch_distance, batch_angles, image_dict):
        waypoint_distances = {}
        waypoint_radius = {}
        waypoint_images = {}

        num_images = len(image_dict)
        interval = 360 / num_images

        for angle_idx, angle in enumerate(batch_angles):
            angle_deg = np.rad2deg(angle) % 360  # 确保在0~360度之间
            # 将角度划入区间 (0, interval], ..., (360 - interval, 360]
            index = int(angle_deg // interval) + 1
            if index == num_images:
                index = 0
            key = str(index)

            if key in image_dict:
                waypoint_images[key] = image_dict[key]
                waypoint_distances[key] = batch_distance[angle_idx]
                waypoint_radius[key] = angle

        return waypoint_images, waypoint_radius, waypoint_distances

    
    def format_observation(self, observation):
        direction_labels = [
            "Front, Angle 0",       # 0°
            "Front Left, Angle 30",     # 30°
            "Front Left, Angle 60",       # 60°
            "Left, Angle 90",     # 90°
            "Rear Left, Angle 120",   # 120°
            "Rear Left, Angle 150",       # 150°
            "Rear, Angle 180",   # 180°
            "Rear Right, Angle 210",     # 210°
            "Rear Right, Angle 240",       # 240°
            "Right, Angle 270",     # 270°
            "Front Right, Angle 300",   # 300°
            "Front Right, Angle 330"    # 330°
        ]

        format_observation = []
        for i, (obs, angle) in enumerate(zip(observation, direction_labels)):
            format_obs = obs.replace(f'Direction Viewpoint ID: {i}', angle)
            format_observation.append(format_obs)
        
        return format_observation

    def concat_images(self, images, rows, columns, padding=10, bg_color=(255, 255, 255), save_path=None):
        assert len(images) <= rows * columns, "图像数量不能超过 rows × columns"

        width, height = images[0].size

        total_width = columns * width + (columns - 1) * padding
        total_height = rows * height + (rows - 1) * padding

        grid_img = Image.new('RGB', (total_width, total_height), color=bg_color)

        for idx, img in enumerate(images):
            x = (idx % columns) * (width + padding)
            y = (idx // columns) * (height + padding)
            grid_img.paste(img, (x, y))

        if save_path:
            grid_img.save(save_path)

        return grid_img

    def safe_remove_keys(self, original_dict, keys_to_remove):
        removed_items = {k: v for k, v in original_dict.items() if k in keys_to_remove}
        modified_dict = {k: v for k, v in original_dict.items() if k not in keys_to_remove}
        if modified_dict:
            return modified_dict, removed_items
        else:
            return original_dict, {}
    
    def compute_absolute_positions(self, pos, heading, angle_dict, distance_dict):

        result = {}
        for pid in angle_dict:
            rel_angle = angle_dict[pid]
            distance = distance_dict[pid]
            
            global_angle = (heading + rel_angle) % (2 * np.pi)
            
            x = pos[0] - distance * np.sin(global_angle)
            y = pos[1]
            z = pos[2] - distance * np.cos(global_angle)
            
            result[pid] = (x, y, z)
        
        return result
    
    def find_candidates_on_path(self, candidate_points_dict, path_points, threshold=0.1):

        ids = list(candidate_points_dict.keys())
        candidate_coords = np.array([candidate_points_dict[i] for i in ids])

        if len(path_points) == 0:
            return []

        tree = cKDTree(path_points)
        dists, _ = tree.query(candidate_coords, k=1)
        matched_ids = [ids[i] for i in range(len(dists)) if dists[i] < threshold]

        return matched_ids
    
    import numpy as np

    def preprocess_depth(self, depth):
        # depth - (B, H, W, 1) numpy array
        DATASET = 'R2R'
        if DATASET == 'R2R':
            min_depth = 0.
            max_depth = 10.
        elif DATASET == 'RxR':
            min_depth = 0.5
            max_depth = 5.0

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max = np.max(depth, axis=1, keepdims=True)  # (B, H, W, 1)
        depth_max = np.tile(depth_max, (1, H, 1, 1))
        depth[depth == 0] = depth_max[depth == 0]

        #mask2 = depth > 0.99
        #depth[mask2] = 0 # noise

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.
        return depth[:,:,:,0]

    def image_get_rel_position(self,depth_map,angle,shape=(112,112)):
        DATASET = 'R2R'
        W=shape[0]
        H=shape[0]
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32)
        if DATASET == 'R2R':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi/4)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi/4.))
        elif DATASET == 'RxR':
            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * 79./360.)
            direction = np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * 79./360.))

        direction = (direction+angle) % (2*math.pi)
        rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        rel_y = -depth_y * math.cos(angle) + depth_x * math.sin(angle)
        rel_z = depth_z
        # rel_x = depth_x * math.cos(angle) + depth_y * math.sin(angle)
        # rel_y = depth_z
        # rel_z = -depth_y * math.cos(angle) + depth_x * math.sin(angle)

        return rel_x, rel_z, rel_y, direction.reshape(-1)

    def getGlobalMap(self, position, heading, depths, shape=(112,112)):
        depth = [cv2.resize(obs, shape, interpolation = cv2.INTER_NEAREST) for obs in depths]
        depth = [depth[0]] + depth[1:][::-1]
        depth = np.stack(depth, 0).reshape([len(depths),shape[0],shape[1],1])
        depth = self.preprocess_depth(depth)
        pcd_x = []
        pcd_y = []
        pcd_z = []
        
        for ix in range(len(depth)):
            dep = depth[ix:ix+1].reshape(-1)
            rel_x, rel_y, rel_z, direction = self.image_get_rel_position(dep, ix*math.pi/(len(depths)/2))  

            rel_x = rel_x[dep < 5]
            rel_y = rel_y[dep < 5]
            rel_z = rel_z[dep < 5]

            image_global_x = rel_x
            image_global_y = rel_y
            image_global_z = rel_z 

            pcd_x.append(image_global_x)
            pcd_y.append(image_global_y)
            pcd_z.append(image_global_z)

        pcd_x = np.concatenate(pcd_x,axis=-1)
        pcd_y = np.concatenate(pcd_y,axis=-1)
        pcd_z = np.concatenate(pcd_z,axis=-1)

        pcd = np.stack([pcd_x, pcd_y, pcd_z], -1)

        return pcd
    
    def add_gaussian_perturbation(self, x, sigma=0.1):
        return x + np.random.normal(0, sigma)

    def _eval_llm(
        self,
        debug=False,
    ) -> None:
        r"""Evaluation.

        Args:
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint

        Returns:
            None
        """
        config = self.config.clone()


        config.defrost()
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        if len(config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                user_input = "yes"  # auto-continue
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
        if config.LOGGER_FILE:
            fname = config.LOGGER_FILE
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                user_input = "yes"  # auto-continue
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
        if config.VIDEO_DIR:
            fname = config.VIDEO_DIR
            if os.path.exists(fname):
                print(f"skipping -- evaluation exists. File path: {fname}")
                user_input = "yes"  # auto-continue
                if user_input != "yes":
                    print("Skipping evaluation.")
                    return
                else:
                    print("Overwriting previous results...")
                

        envs = construct_envs(
            config, get_env_class(config.ENV_NAME),
            auto_reset_done=False,
            episodes_allowed=self.traj
        ) 

        envs.number_of_episodes = [config.TASK_CONFIG.DATASET.EPISODES_TO_LOAD] # set the number of episodes
        dataset_length = sum(envs.number_of_episodes) 
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(config) 
        observation_space = apply_obs_transforms_obs_space(
            envs.observation_spaces[0], obs_transforms
        )
        # self._initialize_policy(
        #     config,
        #     load_from_ckpt=False,
        #     observation_space=observation_space,
        #     action_space=envs.action_spaces[0],
        # )
        # self.policy.eval() 
        # self.waypoint_predictor.eval()
        observations = envs.reset()
        
        instruction, images_list = self.generate_input(observations[-1])
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        ) 
        batch = batch_obs(observations, self.device) 
        batch = apply_obs_transforms_batch(batch, obs_transforms) 

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        ) 

        stats_episodes = {}
        rgb_frames = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        if config.EVAL.EPISODE_COUNT == -1:
            episodes_to_eval = sum(envs.number_of_episodes)
        else:
            episodes_to_eval = min(
                config.EVAL.EPISODE_COUNT, sum(envs.number_of_episodes)
            )

        pbar = tqdm.tqdm(total=episodes_to_eval) if config.use_pbar else None
        log_str = (
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        # set up the logger
        log_file = config.LOGGER_FILE
        if os.path.exists(log_file): os.remove(log_file)
        import logging
        logging.basicConfig(
            format='%(asctime)s - %(filename)s/%(funcName)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt="%Y-%m-%d %H:%M:%S",
            level=os.environ.get("LOGLEVEL", "INFO").upper(),
            stream=sys.stdout,
            filemode="a"
        )
        nav_logger = logging.getLogger("vln_logger")
        nav_logger.addHandler(logging.FileHandler(filename=log_file))
        
        dataset_name = "R2R"
        if not os.path.exists(f"cache_files/{dataset_name}"):
            os.makedirs(f"cache_files/{dataset_name}")

        actions_cache_path = f"./cache_files/{dataset_name}/actions_cache_detail.json"
        if os.path.exists(actions_cache_path): 
            with open(actions_cache_path, "r", encoding="utf-8") as file:
                actions_cache = json.load(file)
        else:
            actions_cache = {} 
        
        navigator = Open_Nav(self.device, config.LLM, config.API_KEY, config.VLM)
        current_step = 0
        nav_history = []
        obs_history = []
        error_number = 0
        vis_positions = []
        action_id = 0
        pertur = 0
        while envs.num_envs > 0 and len(stats_episodes) < episodes_to_eval:
            current_episodes = envs.current_episodes()
            positions = []; headings = []
            for ob_i in range(len(current_episodes)): 
                agent_state_i = envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])
                vis_positions.append(agent_state_i['position'])
            # ==========Navigator start==========
            nav_logger.info(f"==================== The current episode id is {current_episodes[0].episode_id} ====================")
            nav_logger.info("Instruction: "+instruction)
            actions, landmarks = "", ""
            if instruction not in actions_cache.keys():
                actions = navigator.get_actions(instruction)
                landmarks = navigator.get_landmarks(actions)
                actions_cache[instruction] = {"actions": actions, "landmarks": landmarks}
                with open(actions_cache_path, "w", encoding="utf-8") as f2:
                    json.dump(actions_cache, f2, indent=2)
            else:
                actions = actions_cache[instruction]["actions"]
                landmarks = actions_cache[instruction]["landmarks"]
            nav_logger.info("Actions: "+actions)
            nav_logger.info("Landmarks: " + landmarks)
            
            step_length = 6 if len(actions.split("\n")) <= 6 else 8


            stop_flag = False
            current_step += 1
            nav_logger.info(f"-------------------- Step {current_step} --------------------")

            info = envs.get_metrics()

            all_images = [v['rgb'] for k, v in images_list.items()]
            all_images = all_images[:1] + all_images[1::][::-1]
            concat_images = self.concat_images(all_images, 3, 4, save_path="./image_show/concat_rgb.png")

            # SWG
            depths = [observations[0]['depth'][:,:,0]] + [v[:,:,0] for k, v in observations[0].items() if 'depth_' in k]
            pcd = self.getGlobalMap(positions[0], headings[0], depths)
            wp_radius, wp_distance, navi_area = get_structure_wp(pcd, clamp_dist=(1,1.5), map_size=info[0]['top_down_map_vlnce']['map'].shape)
            images_dict, radius_dict, distance_dict = self.construct_image_dicts(wp_distance, wp_radius, images_list)
            cand_pos = self.compute_absolute_positions(positions[0], headings[0], radius_dict, distance_dict)
            matched = self.find_candidates_on_path(cand_pos, np.array(vis_positions[:-1]), threshold=0.5)
            images_dict, _ = self.safe_remove_keys(images_dict, matched)
            radius_dict, _ = self.safe_remove_keys(radius_dict, matched)
            distance_dict, _ = self.safe_remove_keys(distance_dict, matched)
            cand_pos, _ = self.safe_remove_keys(cand_pos, matched)

            if len(config.VIDEO_OPTION) > 0: 
                vis_info = {
                    'nodes': list(vis_positions[1:]),
                    'waypoints': list(cand_pos.values()),
                }
                info = envs.get_metrics()
                frame = observations_to_image(observations[0], info[0], vis_info, items=['td'], navi_area=navi_area)
                Image.fromarray(frame).save(f'waypoint.png')  

            # Reasoning
            if not debug:
                nav_logger.info("========== Get Observation ==========")
                layout = navigator.observe_layout(nav_logger, current_step, concat_images)
                observation, observe_dict = navigator.observe_waypoint(nav_logger, current_step, layout, images_dict)

                nav_logger.info("========== Review History ==========")
                history_traj = navigator.review_history(nav_logger, nav_history) if len(nav_history) > 0 else "Step 0 start position. "

                nav_logger.info("========== Estimate Completion Progress ==========")
                estimation = navigator.estimate_completion(nav_logger, actions, landmarks, history_traj)
            
                if not stop_flag:
                    nav_logger.info("========== Next Action Prediction ==========")
                    predictions, thoughts, break_flag = navigator.move_to_next_vp(nav_logger, current_step, instruction, actions, landmarks, history_traj, estimation, layout, observation, images_dict)

                    nav_logger.info("========== Thought ==========")
                    fused_pred_thought = navigator.thought_fusion(nav_logger, predictions, thoughts)
                    
                    nav_logger.info("========== Test Decision ==========")
                    next_vp, thought, error_number = navigator.test_decisions(nav_logger, fused_pred_thought, observation, instruction, error_number, observe_dict)

            # Action
            if not stop_flag:
                env_actions = []
                env_actions.append({'action':
                    {'action': 4,
                    'action_args':{
                        'angle': radius_dict[next_vp],
                        'distance': distance_dict[next_vp],
                    }}})
                nav_logger.info(f"The final env action: {env_actions}")
                outputs = envs.step(env_actions)     

                previous_image = images_list[next_vp]['rgb']               
    
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]

                if len(config.VIDEO_OPTION) > 0: 
                    vis_info = {
                        'nodes': list(vis_positions[1:]),
                        'waypoints': list(cand_pos.values()),
                    }
                    info = envs.get_metrics()
                    frame = observations_to_image(observations[0], info[0], vis_info, items=['rgb','td'])
                    frame = append_text_to_image(frame, instruction)
                    Image.fromarray(frame).save(f'{config.VIDEO_DIR}/td_{current_episodes[0].episode_id}.png')

                instruction, images_list = self.generate_input(observations[-1])
                error_number = 0 

                # TAR
                next_image = images_list['0']['rgb']
                move_image = self.concat_images([previous_image, next_image], 1, 2, save_path='./image_show/movement.png')
                if not debug:
                    all_actions = [line.strip() for line in actions.strip().split('\n') if line.strip()]
                    if not estimation.strip():
                        executed_actions = []
                    else:
                        executed_actions = [line.strip() for line in estimation.strip().split('\n') if line.strip()]
                    current_index = len(executed_actions)
                    if current_index < len(all_actions):
                        current_action = all_actions[current_index]
                    else:
                        current_action = all_actions[-1]
                    executed, flag = navigator.action_completion(nav_logger, current_action, move_image)
                    thought += executed

                if not debug:
                    curr_observe = observe_dict[next_vp]
                    nav_logger.info("========== save history ==========")
                    nav_history = navigator.save_history(nav_logger, current_step, next_vp, thought, curr_observe, nav_history)

                # finish navigation
                if current_step == step_length:
                    dones[0] = True 
                else:
                    for j, ob in enumerate(observations):
                        envs.call_at(j, 
                            'change_current_path',
                            {'new_path': ob.pop('positions'),
                            'collisions': ob.pop('collisions')}
                        )
            else:
                dones[0] = True
            
            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8, device=self.device)
            
            for i in range(envs.num_envs):
                
                if not dones[i]:
                    continue
                
                current_step = 0
                nav_history = []
                obs_history = []
                action_id = 0
                info = infos[i]
                metric = {}
                metric['steps_taken'] = info['steps_taken']
                ep_id = str(envs.current_episodes()[i].episode_id)
                gt_path = np.array(self.gt_data[ep_id]['locations']).astype(float)
                if 'current_path' in envs.current_episodes()[i].info.keys():
                    positions_ = np.array(envs.current_episodes()[i].info['current_path']).astype(float)
                    collisions_ = np.array(envs.current_episodes()[i].info['collisions'])
                    assert collisions_.shape[0] == positions_.shape[0] - 1
                else:
                    positions_ = np.array(dis_to_con(np.array(info['position']['position']))).astype(float)
                distance = np.array(info['position']['distance']).astype(float)
                metric['distance_to_goal'] = distance[-1]
                metric['success'] = 1. if distance[-1] <= 3. else 0.
                metric['oracle_success'] = 1. if (distance <= 3.).any() else 0.
                metric['path_length'] = np.linalg.norm(positions_[1:] - positions_[:-1],axis=1).sum()
                metric['collisions'] = collisions_.mean()
                gt_length = distance[0]
                metric['spl'] = metric['success']*gt_length/max(gt_length,metric['path_length'])

                act_con_path = positions_
                gt_con_path = np.array(gt_path).astype(float)
                dtw_distance = fastdtw(act_con_path, gt_con_path, dist=NDTW.euclidean_distance)[0]
                nDTW = np.exp(-dtw_distance / (len(gt_con_path) * config.TASK_CONFIG.TASK.SUCCESS_DISTANCE))

                metric['ndtw'] = nDTW
                stats_episodes[current_episodes[i].episode_id] = metric 
                nav_logger.info(metric)

                observations[i] = envs.reset_at(i)[0]
                instruction, images_list = self.generate_input(observations[i])

                vis_positions = []
                
                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=episodes_to_eval,
                            time=round(time.time() - start_time),
                        )
                    )
            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, obs_transforms)   
            
            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            headings = torch.tensor(headings)
            (
                envs,
                not_done_masks,
                headings,  
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                not_done_masks,
                headings,
                batch,
                rgb_frames,
            )
            headings = headings.tolist()
            # except Exception as e:
            #     nav_logger.info(f"Error in next action prediction: {e}")
            #     current_step -= 1

        envs.close()
        if config.use_pbar:
            pbar.close()
        if self.world_size > 1:
            distr.barrier()
        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )
        total = torch.tensor(num_episodes).cuda()
        if self.world_size > 1:
            dist.reduce(total,dst=0)
        total = total.item()

        if self.world_size > 1:
            logger.info(
                f"rank {self.local_rank}'s {num_episodes}-episode results: {aggregated_stats}")
            for k,v in aggregated_stats.items():
                v = torch.tensor(v*num_episodes).cuda()
                cat_v = gather_list_and_concat(v,self.world_size)
                v = (sum(cat_v)/total).item()
                aggregated_stats[k] = v

        split = config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            config.RESULTS_DIR,
            f"stats_ep_ckpt_{split}_r{self.local_rank}_w{self.world_size}.json",
        )
        with open(fname, "w") as f:
            json.dump(stats_episodes, f, indent=4)

        if self.local_rank < 1:
            if config.EVAL.SAVE_RESULTS:
                fname = os.path.join(
                    config.RESULTS_DIR,
                    f"stats_ckpt_{split}.json",
                )
                with open(fname, "w") as f:
                    json.dump(aggregated_stats, f, indent=4)

            logger.info(f"Episodes evaluated: {total}")
            for k, v in aggregated_stats.items():
                logger.info(f"Average episode {k}: {v:.6f}")
        
    def collect_val_traj(self):
        trajectories = defaultdict(list)
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        with gzip.open(
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                split=split)
        ) as f:
            gt_data = json.load(f)
        self.gt_data = gt_data
        trajectories = gt_data
        self.trajectories = gt_data
        trajectories = list(trajectories.keys())[self.config.local_rank::self.config.GPU_NUMBERS]
        random.shuffle(trajectories)
        # Apply cross-floor filter if EPISODES_ALLOWED is explicitly set
        allowed = self.config.TASK_CONFIG.DATASET.EPISODES_ALLOWED
        if allowed is not None:
            allowed_set = set(allowed)
            # trajectories are always str (JSON dict keys); allowed may be int or str
            trajectories = [t for t in trajectories
                            if t in allowed_set or int(t) in allowed_set]
        return trajectories
        
    def eval(self) -> None:
        r"""Main method of trainer evaluation. 

        Returns:
            None
        """
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        if "tensorboard" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.TENSORBOARD_DIR) > 0
            ), "Must specify a tensorboard directory for video display"
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        if "disk" in self.config.VIDEO_OPTION:
            assert (
                len(self.config.VIDEO_DIR) > 0
            ), "Must specify a directory for storing videos on disk"

        world_size = self.config.GPU_NUMBERS
        self.world_size = world_size
        self.local_rank = self.config.local_rank

        self.config.defrost()
        self.config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        self.config.TASK_CONFIG.TASK.MEASUREMENTS = ['POSITION',
                                                     'STEPS_TAKEN',
                                                     ]
        if 'HIGHTOLOW' in self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS:
            idx = self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS.index('HIGHTOLOW')
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS[idx] = 'HIGHTOLOWEVAL'
        self.config.TASK_CONFIG.DATASET.LANGUAGES = self.config.EVAL.LANGUAGES
        self.config.TASK_CONFIG.DATASET.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.config.EVAL.SPLIT
        self.config.use_pbar = not is_slurm_batch_job()
        if 'rxr' in self.config.BASE_TASK_CONFIG_PATH:
            self.config.EVAL.trajectories_file = \
                self.config.EVAL.trajectories_file[:-8] + '_w' + \
                str(self.world_size) + '_r' + str(self.local_rank) + '.json.gz'
        
        # if choosing image
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations(12)

        # sensor_uuids = []
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            sensor = getattr(config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                # sensor_uuids.append(camera_config.UUID)
                setattr(config.SIMULATOR, camera_template, camera_config)
                config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.TASK_CONFIG = config
        self.config.SENSORS = config.SIMULATOR.AGENT_0.SENSORS
        
        self.config.freeze()
        torch.cuda.set_device(self.device)
        if world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            torch.cuda.set_device(self.device)
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            
        self.traj = self.collect_val_traj()
        debug = False
        self._eval_llm(debug)

