# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import numpy as np
import torch

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from .base_client import InferenceClient


class SmartWorldDroidJointposClient(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 7777,
        open_loop_horizon: int = 8,
    ) -> None:
        print(f"[{self.__class__.__name__}] Awaiting server on {remote_host}:{remote_port}...")
        self.client = WebsocketClientPolicy(
            remote_host,
            remote_port,
            ping_interval=300.0,
            ping_timeout=300.0,
        )
        print(f"[{self.__class__.__name__}] Server metadata: {self.client.get_server_metadata()}")

        self.open_loop_horizon = int(open_loop_horizon)
        self._env_chunk: dict[int, np.ndarray] = {}
        self._env_counter: dict[int, int] = {}
        self._env_control_step: dict[int, int] = {}
        self._env_executed_actions: dict[int, list[np.ndarray]] = {}
        self._env_image_history: dict[int, list[tuple[int, np.ndarray]]] = {}

    def reset(self):
        self._env_chunk.clear()
        self._env_counter.clear()
        self._env_control_step.clear()
        self._env_executed_actions.clear()
        self._env_image_history.clear()
        self.client.predict_action({"reset": True})

    def close(self):
        self.client.close()

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        curr_obs = self._extract_observation(obs, env_id=env_id)

        if env_id in self._env_counter:
            counter = self._env_counter[env_id]
        else:
            counter = 0
        if env_id in self._env_control_step:
            control_step = self._env_control_step[env_id]
        else:
            control_step = 0
        if control_step > 0:
            executed_action = np.concatenate([
                curr_obs["joint_position"].reshape(7),
                curr_obs["gripper_position"].reshape(1),
            ]).astype(np.float32)
            if env_id not in self._env_executed_actions:
                self._env_executed_actions[env_id] = []
            self._env_executed_actions[env_id].append(executed_action)

        stitched_frame = np.concatenate([curr_obs["external_image_0"], curr_obs["external_image_1"], curr_obs["wrist_image"]], axis=1)
        if env_id not in self._env_image_history:
            self._env_image_history[env_id] = []
        needs_query = counter == 0 or counter >= self.open_loop_horizon or env_id not in self._env_chunk
        if needs_query:
            executed_count = counter
            if env_id in self._env_executed_actions:
                executed_actions = self._env_executed_actions[env_id]
            else:
                executed_actions = []
            counter = 0
            request_data = {
                "observation/exterior_image_0_left": curr_obs["external_image_0"],
                "observation/exterior_image_1_left": curr_obs["external_image_1"],
                "observation/wrist_image_left": curr_obs["wrist_image"],
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
                "control_step": control_step,
                "history/executed_action_count": executed_count,
            }
            if len(executed_actions) > 0:
                request_data["history/executed_actions"] = np.stack(executed_actions, axis=0).astype(np.float32)
            image_history = self._env_image_history[env_id]
            if len(image_history) > 0:
                history_steps, history_frames = zip(*image_history)
                request_data["history/step_indices"] = np.asarray(history_steps, dtype=np.int64)
                request_data["history/stitched_frames"] = np.stack(history_frames, axis=0).astype(np.uint8, copy=False)
            response = self.client.predict_action(request_data)
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"SmartWorld server must return action chunk [T,D], got shape={actions.shape}.")
            self._env_chunk[env_id] = actions
            self._env_executed_actions[env_id] = []
            self._env_image_history[env_id] = []
        else:
            self._env_image_history[env_id].append((int(control_step), stitched_frame.copy()))

        action = self._env_chunk[env_id][counter].copy()
        self._env_counter[env_id] = counter + 1
        self._env_control_step[env_id] = control_step + 1

        if action.shape != (8,):
            raise ValueError(f"SmartWorld RoboLab client expects 8D action, got shape={action.shape}.")
        action[-1] = 1.0 if action[-1] > 0.5 else 0.0

        return {"action": action, "viz": stitched_frame}

    def _extract_observation(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        external_image_0 = self._tensor_image_to_numpy(obs_dict["image_obs"]["external_cam"][env_id])
        external_image_1 = self._tensor_image_to_numpy(obs_dict["image_obs"]["external_cam_2"][env_id])
        wrist_image = self._tensor_image_to_numpy(obs_dict["image_obs"]["wrist_cam"][env_id])

        robot_state = obs_dict["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)

        if joint_position.shape != (7,):
            raise ValueError(f"Expected 7D joint_position, got shape={joint_position.shape}.")
        if gripper_position.reshape(-1).shape != (1,):
            raise ValueError(f"Expected 1D gripper_position, got shape={gripper_position.shape}.")

        return {
            "external_image_0": external_image_0,
            "external_image_1": external_image_1,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position.reshape(1),
        }

    @staticmethod
    def _tensor_image_to_numpy(image: torch.Tensor) -> np.ndarray:
        array = image.clone().detach().cpu().numpy()
        if array.ndim != 3:
            raise ValueError(f"Expected image [H,W,C], got shape={array.shape}.")
        if array.shape[2] != 3:
            raise ValueError(f"Expected image channel dimension 3, got shape={array.shape}.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
