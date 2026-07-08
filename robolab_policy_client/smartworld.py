# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os
import time

import msgpack
import numpy as np
import torch
import torch.nn.functional as F
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
        self._profile_timing = os.environ.get("SMARTWORLD_PROFILE_CLIENT_TIMING", "0").strip().lower() in {"1", "true", "yes", "on"}
        self._request_index = 0

    def get_server_metadata(self) -> dict:
        return dict(self._metadata)

    def predict_action(self, request: dict) -> dict:
        self._request_index += 1
        total_start = time.perf_counter()
        pack_start = time.perf_counter()
        packed_request = self._packer.pack(request)
        pack_sec = time.perf_counter() - pack_start
        send_start = time.perf_counter()
        self._ws.send(packed_request)
        send_sec = time.perf_counter() - send_start
        recv_start = time.perf_counter()
        packed_response = self._ws.recv()
        recv_sec = time.perf_counter() - recv_start
        unpack_start = time.perf_counter()
        response = self._packer.unpack(packed_response)
        unpack_sec = time.perf_counter() - unpack_start
        if not isinstance(response, dict):
            raise TypeError(f"Expected server response dict, got {type(response)!r}.")
        if self._profile_timing:
            total_sec = time.perf_counter() - total_start
            print(
                f"[SmartWorldWebsocketClient] timing request_index={self._request_index} "
                f"request_bytes={len(packed_request)} response_bytes={len(packed_response)} "
                f"total={total_sec:.3f}s pack={pack_sec:.3f}s send={send_sec:.3f}s "
                f"recv={recv_sec:.3f}s unpack={unpack_sec:.3f}s",
                flush=True,
            )
        return response

    def close(self) -> None:
        self._ws.close()


class SmartWorldDroidJointposClient(InferenceClient):
    requires_dedicated_env_client = True
    _CAMERA_REQUEST_KEYS = (
        "observation/exterior_image_0_left",
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
    )

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 7777,
        open_loop_horizon: int | None = None,
        experiment_name: str | None = None,
        return_viz: bool = True,
    ) -> None:
        self.remote_host = remote_host
        self.remote_port = int(remote_port)
        self.open_loop_horizon = None if open_loop_horizon is None else int(open_loop_horizon)
        self.experiment_name = experiment_name
        self.return_viz = bool(return_viz)
        self.image_height = int(os.environ.get("SMARTWORLD_CLIENT_IMAGE_HEIGHT", "320"))
        self.image_width = int(os.environ.get("SMARTWORLD_CLIENT_IMAGE_WIDTH", "480"))

        print(f"[{self.__class__.__name__}] Awaiting server on {remote_host}:{remote_port}...")
        self.client = SmartWorldWebsocketClient(
            remote_host,
            remote_port,
        )
        self._server_metadata = self.client.get_server_metadata()
        if "image_height" in self._server_metadata or "image_width" in self._server_metadata:
            if "image_height" not in self._server_metadata or "image_width" not in self._server_metadata:
                raise ValueError(
                    "SmartWorld server metadata must provide both image_height and image_width when overriding "
                    "client image size."
                )
            self.image_height = int(self._server_metadata["image_height"])
            self.image_width = int(self._server_metadata["image_width"])
        self._requires_image_history = self._metadata_requires_image_history(self._server_metadata)
        print(f"[{self.__class__.__name__}] Server metadata: {self._server_metadata}")
        print(f"[{self.__class__.__name__}] Sending image size: {self.image_height}x{self.image_width}")
        if bool(self._server_metadata.get("causal_action_rollout")) and self.open_loop_horizon is not None:
            causal_chunk_len = int(self._server_metadata.get("causal_action_chunk_len") or 0)
            if causal_chunk_len > 0 and int(self.open_loop_horizon) < causal_chunk_len:
                raise ValueError(
                    "SmartWorld causal rollout requires consuming a full causal action chunk before re-querying. "
                    f"Got open_loop_horizon={self.open_loop_horizon}, causal_action_chunk_len={causal_chunk_len}."
                )
        self._camera_request_keys = self._resolve_camera_request_keys(self._server_metadata)
        print(f"[{self.__class__.__name__}] Sending camera keys: {self._camera_request_keys}")
        print(f"[{self.__class__.__name__}] Sending image history: {self._requires_image_history}")

        self._env_chunk: dict[int, np.ndarray] = {}
        self._env_counter: dict[int, int] = {}
        self._env_step: dict[int, int] = {}
        self._env_executed_actions: dict[int, list[np.ndarray]] = {}
        self._env_image_history: dict[int, list[tuple[int, dict[str, np.ndarray]]]] = {}
        self._profile_sections: list[tuple[str, float]] = []

    def _resolve_camera_request_keys(self, metadata: dict | None = None) -> list[str]:
        metadata = self._server_metadata if metadata is None else metadata
        raw_keys = metadata.get("required_camera_keys")
        if raw_keys is None:
            keys_by_purpose = metadata.get("camera_request_keys_by_purpose")
            if keys_by_purpose is None:
                keys_by_purpose = metadata.get("camera_keys_by_purpose")
            if isinstance(keys_by_purpose, dict):
                flattened = []
                for purpose_keys in keys_by_purpose.values():
                    if isinstance(purpose_keys, (list, tuple)):
                        flattened.extend(purpose_keys)
                raw_keys = flattened
        if raw_keys is None:
            return list(self._CAMERA_REQUEST_KEYS)
        if not isinstance(raw_keys, (list, tuple)):
            raise TypeError(f"Server required_camera_keys must be a list, got {type(raw_keys)!r}.")
        selected = []
        for raw_key in raw_keys:
            key = str(raw_key)
            if key not in self._CAMERA_REQUEST_KEYS:
                raise ValueError(
                    f"SmartWorld server requested unsupported camera key {key!r}. "
                    f"Supported keys: {list(self._CAMERA_REQUEST_KEYS)!r}."
                )
            if key not in selected:
                selected.append(key)
        return selected or list(self._CAMERA_REQUEST_KEYS)

    def clone(self) -> "SmartWorldDroidJointposClient":
        return type(self)(
            remote_host=self.remote_host,
            remote_port=self.remote_port,
            open_loop_horizon=self.open_loop_horizon,
            experiment_name=self.experiment_name,
            return_viz=self.return_viz,
        )

    def reset(self, *, env_id: int | None = None) -> None:
        request = {"reset": True}
        if self.experiment_name is not None:
            request["experiment_name"] = self.experiment_name
        response = self.client.predict_action(request)
        if isinstance(response, dict):
            self._server_metadata.update(response)
            self._requires_image_history = self._metadata_requires_image_history(self._server_metadata)
            self._camera_request_keys = self._resolve_camera_request_keys(self._server_metadata)
        self._profile_sections.clear()
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
        counter = self._env_counter.get(env_id, 0)
        control_step = self._env_step.get(env_id, 0)
        needs_query = counter == 0 or env_id not in self._env_chunk
        if not needs_query:
            chunk_horizon = int(self._env_chunk[env_id].shape[0])
            refresh_horizon = chunk_horizon if self.open_loop_horizon is None else min(self.open_loop_horizon, chunk_horizon)
            needs_query = counter >= refresh_horizon

        state_start = time.perf_counter()
        curr_state = self._extract_state(obs, env_id=env_id)
        self._record_profile("smartworld_extract_state", time.perf_counter() - state_start)

        if control_step > 0:
            executed_action = np.concatenate([
                curr_state["joint_position"].reshape(7),
                curr_state["gripper_position"].reshape(1),
            ]).astype(np.float32)
            if env_id not in self._env_executed_actions:
                self._env_executed_actions[env_id] = []
            self._env_executed_actions[env_id].append(executed_action)
        if env_id not in self._env_image_history:
            self._env_image_history[env_id] = []

        process_images = bool(needs_query or self._requires_image_history or self.return_viz)
        images = None
        all_views = None
        history_views = None
        if process_images:
            images = self._extract_resized_images(obs, env_id=env_id)
            all_views = {
                "observation/exterior_image_0_left": images["external_image_0"],
                "observation/exterior_image_1_left": images["external_image_1"],
                "observation/wrist_image_left": images["wrist_image"],
            }
            history_views = {key: all_views[key] for key in self._camera_request_keys}

        if needs_query:
            if images is None or all_views is None or history_views is None:
                images = self._extract_resized_images(obs, env_id=env_id)
                all_views = {
                    "observation/exterior_image_0_left": images["external_image_0"],
                    "observation/exterior_image_1_left": images["external_image_1"],
                    "observation/wrist_image_left": images["wrist_image"],
                }
                history_views = {key: all_views[key] for key in self._camera_request_keys}
            executed_count = counter
            counter = 0
            executed_actions = self._env_executed_actions.get(env_id, [])
            request_data = {
                "observation/joint_position": curr_state["joint_position"],
                "observation/gripper_position": curr_state["gripper_position"],
                "prompt": instruction,
                "control_step": control_step,
                "history/executed_action_count": executed_count,
            }
            request_data.update({key: all_views[key] for key in self._camera_request_keys})
            if len(executed_actions) > 0:
                request_data["history/executed_actions"] = np.stack(executed_actions, axis=0).astype(np.float32)
            image_history = self._env_image_history[env_id]
            if self._requires_image_history and len(image_history) > 0:
                stack_start = time.perf_counter()
                history_steps, history_view_dicts = zip(*image_history)
                request_data["history/step_indices"] = np.asarray(history_steps, dtype=np.int64)
                for request_key in history_views:
                    request_data[f"history/{request_key}"] = np.stack(
                        [view_dict[request_key] for view_dict in history_view_dicts],
                        axis=0,
                    ).astype(np.uint8, copy=False)
                self._record_profile("smartworld_history_stack", time.perf_counter() - stack_start)
            websocket_start = time.perf_counter()
            response = self.client.predict_action(request_data)
            self._record_profile("smartworld_websocket", time.perf_counter() - websocket_start)
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2:
                raise ValueError(f"SmartWorld server must return action chunk [T,D], got shape={actions.shape}.")
            self._env_chunk[env_id] = actions
            self._env_executed_actions[env_id] = []
            self._env_image_history[env_id] = []
        elif self._requires_image_history:
            if history_views is None:
                raise RuntimeError("SmartWorld image history is required but images were not prepared.")
            copy_start = time.perf_counter()
            self._env_image_history[env_id].append(
                (
                    int(control_step),
                    {key: value.copy() for key, value in history_views.items()},
                )
            )
            self._record_profile("smartworld_history_copy", time.perf_counter() - copy_start)

        action = self._env_chunk[env_id][counter].copy()
        self._env_counter[env_id] = counter + 1
        self._env_step[env_id] = control_step + 1

        if action.shape != (8,):
            raise ValueError(f"SmartWorld RoboLab client expects 8D action, got shape={action.shape}.")

        result = {"action": action}
        if self.return_viz and images is not None:
            viz_start = time.perf_counter()
            result["viz"] = np.concatenate(
                [
                    images["external_image_0"],
                    images["wrist_image"],
                    images["external_image_1"],
                ],
                axis=1,
            )
            self._record_profile("smartworld_viz_concat", time.perf_counter() - viz_start)
        return result

    def consume_profile_sections(self) -> list[tuple[str, float]]:
        sections = self._profile_sections
        self._profile_sections = []
        return sections

    def _record_profile(self, name: str, elapsed: float) -> None:
        self._profile_sections.append((name, float(elapsed)))

    @staticmethod
    def _metadata_requires_image_history(metadata: dict) -> bool:
        if "requires_image_history" not in metadata:
            return True
        return bool(metadata["requires_image_history"])

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        views = {
            "observation/exterior_image_0_left": self._resize_image(
                extracted_obs["external_image_0"], self.image_height, self.image_width
            ),
            "observation/exterior_image_1_left": self._resize_image(
                extracted_obs["external_image_1"], self.image_height, self.image_width
            ),
            "observation/wrist_image_left": self._resize_image(
                extracted_obs["wrist_image"], self.image_height, self.image_width
            ),
        }
        request = {
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }
        request.update({key: views[key] for key in self._camera_request_keys})
        return request

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
        return {
            **self._extract_resized_images(obs_dict, env_id=env_id),
            **self._extract_state(obs_dict, env_id=env_id),
        }

    def _extract_state(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        robot_state = obs_dict["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)

        if joint_position.shape != (7,):
            raise ValueError(f"Expected 7D joint_position, got shape={joint_position.shape}.")
        if gripper_position.reshape(-1).shape != (1,):
            raise ValueError(f"Expected 1D gripper_position, got shape={gripper_position.shape}.")

        return {
            "joint_position": joint_position,
            "gripper_position": gripper_position.reshape(1),
        }

    def _extract_resized_images(self, obs_dict: dict, *, env_id: int = 0) -> dict:
        image_obs = obs_dict["image_obs"]
        image_keys = ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
        missing_image_keys = [key for key in image_keys if key not in image_obs]
        if missing_image_keys:
            raise KeyError(
                "SmartWorld requires DROID left/right/wrist cameras; "
                f"missing={missing_image_keys}, available={list(image_obs)}."
            )

        resize_start = time.perf_counter()
        tensors = [image_obs[key][env_id].detach() for key in image_keys]
        for tensor in tensors:
            if tensor.ndim != 3:
                raise ValueError(f"Expected image [H,W,C], got shape={tuple(tensor.shape)}.")
            if int(tensor.shape[2]) != 3:
                raise ValueError(f"Expected image channel dimension 3, got shape={tuple(tensor.shape)}.")
        batched = torch.stack(tensors, dim=0)
        original_dtype = batched.dtype
        if int(batched.shape[1]) != self.image_height or int(batched.shape[2]) != self.image_width:
            batched = batched.permute(0, 3, 1, 2).float()
            batched = F.interpolate(
                batched,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )
            batched = batched.clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        elif original_dtype != torch.uint8:
            batched = batched.clamp(0, 255).to(torch.uint8).contiguous()
        else:
            batched = batched.contiguous()
        self._record_profile("smartworld_resize", time.perf_counter() - resize_start)

        copy_start = time.perf_counter()
        arrays = batched.cpu().numpy()
        self._record_profile("smartworld_image_gpu_to_cpu", time.perf_counter() - copy_start)

        return {
            "external_image_0": arrays[0],
            "external_image_1": arrays[1],
            "wrist_image": arrays[2],
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
