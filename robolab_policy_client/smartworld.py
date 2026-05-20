# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import msgpack
import numpy as np
import torch
import websockets.sync.client
from PIL import Image

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
    ) -> None:
        self._uri = f"ws://{remote_host}:{remote_port}"
        self._packer = MsgPackNumpy()
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
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
    requires_dedicated_env_client = True

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 7777,
        open_loop_horizon: int = 8,
    ) -> None:
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.open_loop_horizon = int(open_loop_horizon)
        self.image_height = 180
        self.image_width = 320

        print(f"[{self.__class__.__name__}] Awaiting server on {remote_host}:{remote_port}...")
        self.client = SmartWorldWebsocketClient(
            remote_host,
            remote_port,
        )
        print(f"[{self.__class__.__name__}] Server metadata: {self.client.get_server_metadata()}")

        self._env_chunk: dict[int, np.ndarray] = {}
        self._env_counter: dict[int, int] = {}
        self._env_step: dict[int, int] = {}
        self._env_executed_actions: dict[int, list[np.ndarray]] = {}
        self._env_image_history: dict[int, list[tuple[int, np.ndarray]]] = {}

    def clone(self) -> "SmartWorldDroidJointposClient":
        return type(self)(
            remote_host=self.remote_host,
            remote_port=self.remote_port,
            open_loop_horizon=self.open_loop_horizon,
        )

    def reset(self, *, env_id: int | None = None) -> None:
        self.client.predict_action({"reset": True})
        if env_id is None:
            self._env_chunk.clear()
            self._env_counter.clear()
            self._env_step.clear()
            self._env_executed_actions.clear()
            self._env_image_history.clear()
            return
        self._env_chunk.pop(env_id, None)
        self._env_counter.pop(env_id, None)
        self._env_step.pop(env_id, None)
        if env_id in self._env_executed_actions:
            self._env_executed_actions.pop(env_id)
        if env_id in self._env_image_history:
            self._env_image_history.pop(env_id)

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
        if control_step > 0:
            executed_action = np.concatenate([
                curr_obs["joint_position"].reshape(7),
                curr_obs["gripper_position"].reshape(1),
            ]).astype(np.float32)
            if env_id not in self._env_executed_actions:
                self._env_executed_actions[env_id] = []
            self._env_executed_actions[env_id].append(executed_action)
        external_image_0 = self._resize_image(curr_obs["external_image_0"], self.image_height, self.image_width)
        external_image_1 = self._resize_image(curr_obs["external_image_1"], self.image_height, self.image_width)
        wrist_image = self._resize_image(curr_obs["wrist_image"], self.image_height, self.image_width)
        stitched_frame = np.concatenate([external_image_0, external_image_1, wrist_image], axis=1)
        if env_id not in self._env_image_history:
            self._env_image_history[env_id] = []
        needs_query = counter == 0 or counter >= self.open_loop_horizon or env_id not in self._env_chunk
        if needs_query:
            executed_count = counter
            counter = 0
            if env_id in self._env_executed_actions:
                executed_actions = self._env_executed_actions[env_id]
            else:
                executed_actions = []
            request_data = {
                "observation/exterior_image_0_left": external_image_0,
                "observation/exterior_image_1_left": external_image_1,
                "observation/wrist_image_left": wrist_image,
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
        self._env_step[env_id] = control_step + 1

        if action.shape != (8,):
            raise ValueError(f"SmartWorld RoboLab client expects 8D action, got shape={action.shape}.")
        action[-1] = 1.0 if action[-1] > 0.5 else 0.0

        viz_images = [
            external_image_0,
            wrist_image,
            external_image_1,
        ]
        viz = np.concatenate(viz_images, axis=1)
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

    @staticmethod
    def _resize_image(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        if int(image.shape[0]) == int(target_height) and int(image.shape[1]) == int(target_width):
            return image
        return np.asarray(Image.fromarray(image).resize((target_width, int(target_height))))

    def _extract_observation(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        image_obs = obs_dict["image_obs"]
        required_image_keys = ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
        missing_image_keys = [key for key in required_image_keys if key not in image_obs]
        if missing_image_keys:
            raise KeyError(
                "SmartWorld requires DROID left/right/wrist cameras; "
                f"missing={missing_image_keys}, available={list(image_obs)}."
            )

        external_image_0 = self._tensor_image_to_numpy(image_obs["over_shoulder_left_camera"][env_id])
        external_image_1 = self._tensor_image_to_numpy(image_obs["over_shoulder_right_camera"][env_id])
        wrist_image = self._tensor_image_to_numpy(image_obs["wrist_cam"][env_id])

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
