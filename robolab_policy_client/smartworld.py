# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import msgpack
import numpy as np
import torch
import websockets.sync.client

from robolab.eval.base_client import InferenceClient


class MsgPackNumpy:
    def pack(self, obj):
        return msgpack.packb(obj, default=self._encode_numpy)

    def unpack(self, data):
        return msgpack.unpackb(data, object_hook=self._decode_numpy, strict_map_key=False)

    def _encode_numpy(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported dtype: {obj.dtype}")
            return {
                b"__ndarray__": True,
                b"data": obj.tobytes(),
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }

        if isinstance(obj, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": obj.item(),
                b"dtype": obj.dtype.str,
            }

        return obj

    def _decode_numpy(self, obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        if b"__npgeneric__" in obj:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj


class SmartWorldWebsocketClient:
    def __init__(
        self,
        remote_host: str,
        remote_port: int,
        *,
        ping_interval: float,
        ping_timeout: float,
    ) -> None:
        self._uri = f"ws://{remote_host}:{remote_port}"
        self._packer = MsgPackNumpy()
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
        )
        metadata_bytes = self._ws.recv()
        self._metadata = self._packer.unpack(metadata_bytes)
        if not isinstance(self._metadata, dict):
            raise TypeError(f"Expected server metadata dict, got {type(self._metadata)!r}.")

    def get_server_metadata(self) -> dict:
        return dict(self._metadata)

    def predict_action(self, request: dict) -> dict:
        self._ws.send(self._packer.pack(request))
        response = self._packer.unpack(self._ws.recv())
        if not isinstance(response, dict):
            raise TypeError(f"Expected server response dict, got {type(response)!r}.")
        return response

    def close(self) -> None:
        self._ws.close()


class SmartWorldDroidJointposClient(InferenceClient):
    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 7777,
        open_loop_horizon: int = 8,
    ) -> None:
        print(f"[{self.__class__.__name__}] Awaiting server on {remote_host}:{remote_port}...")
        self.client = SmartWorldWebsocketClient(
            remote_host,
            remote_port,
            ping_interval=300.0,
            ping_timeout=300.0,
        )
        print(f"[{self.__class__.__name__}] Server metadata: {self.client.get_server_metadata()}")

        self.open_loop_horizon = int(open_loop_horizon)
        self._env_chunk: dict[int, np.ndarray] = {}
        self._env_counter: dict[int, int] = {}
        self._env_step: dict[int, int] = {}
        self._env_history_steps: dict[int, list[int]] = {}
        self._env_history_frames: dict[int, list[np.ndarray]] = {}

    def reset(self, *, env_id: int | None = None) -> None:
        self.client.predict_action({"reset": True})
        if env_id is None:
            self._env_chunk.clear()
            self._env_counter.clear()
            self._env_step.clear()
            self._env_history_steps.clear()
            self._env_history_frames.clear()
            return
        self._env_chunk.pop(env_id, None)
        self._env_counter.pop(env_id, None)
        self._env_step.pop(env_id, None)
        self._env_history_steps.pop(env_id, None)
        self._env_history_frames.pop(env_id, None)

    def close(self) -> None:
        self.client.close()

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        curr_obs = self._extract_observation(obs, env_id=env_id)

        if env_id in self._env_counter:
            counter = self._env_counter[env_id]
        else:
            counter = 0
        if env_id in self._env_step:
            control_step = self._env_step[env_id]
        else:
            control_step = 0
        if env_id not in self._env_history_steps:
            self._env_history_steps[env_id] = []
        if env_id not in self._env_history_frames:
            self._env_history_frames[env_id] = []
        stitched_frame = np.concatenate(
            [curr_obs["external_image_0"], curr_obs["external_image_1"], curr_obs["wrist_image"]],
            axis=0,
        )

        needs_query = counter == 0 or counter >= self.open_loop_horizon or env_id not in self._env_chunk
        if needs_query:
            executed_count = counter
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
                "history/step_indices": np.asarray(self._env_history_steps[env_id], dtype=np.int64),
                "history/stitched_frames": list(self._env_history_frames[env_id]),
            }
            response = self.client.predict_action(request_data)
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"SmartWorld server must return action chunk [T,D], got shape={actions.shape}.")
            self._env_chunk[env_id] = actions

        action = self._env_chunk[env_id][counter].copy()
        self._env_counter[env_id] = counter + 1
        self._env_history_steps[env_id].append(control_step)
        self._env_history_frames[env_id].append(stitched_frame)
        if len(self._env_history_steps[env_id]) > 256:
            self._env_history_steps[env_id] = self._env_history_steps[env_id][-256:]
            self._env_history_frames[env_id] = self._env_history_frames[env_id][-256:]
        self._env_step[env_id] = control_step + 1

        if action.shape != (8,):
            raise ValueError(f"SmartWorld RoboLab client expects 8D action, got shape={action.shape}.")
        action[-1] = 1.0 if action[-1] > 0.5 else 0.0

        viz = np.concatenate([curr_obs["external_image_0"], curr_obs["external_image_1"], curr_obs["wrist_image"]], axis=1)
        return {"action": action, "viz": viz}

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "observation/exterior_image_0_left": extracted_obs["external_image_0"],
            "observation/exterior_image_1_left": extracted_obs["external_image_1"],
            "observation/wrist_image_left": extracted_obs["wrist_image"],
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        return self.client.predict_action(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        actions = np.asarray(response["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"SmartWorld server must return action chunk [T,D], got shape={actions.shape}.")
        return actions

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
