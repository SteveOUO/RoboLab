# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import math
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class Cosmos3Client(InferenceClient):
    """RoboLab client for the official Cosmos3-Nano-Policy-DROID server."""

    IMAGE_W = 640
    IMAGE_H = 360
    _SUPPORTED_CAMERA_KEYS = {
        "observation/exterior_image_1_left": "left_image",
        "observation/exterior_image_2_left": "right_image",
        "observation/wrist_image_left": "wrist_image",
    }

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int | None = None,
        remote_uri: str | None = None,
    ) -> None:
        super().__init__()
        self._profile_sections: list[tuple[str, float]] = []
        self._remote_host = remote_host
        self._remote_port = int(remote_port)
        self._remote_uri = remote_uri
        self.open_loop_horizon = None if open_loop_horizon is None else int(open_loop_horizon)
        if self.open_loop_horizon is not None and self.open_loop_horizon < 1:
            raise ValueError(f"open_loop_horizon must be positive, got {self.open_loop_horizon}.")

        self._image_w = self.IMAGE_W
        self._image_h = self.IMAGE_H
        self._stateful_action_rollout = False
        self._rollout_chunk_len = 0
        self._rollout_execute_horizon = 0
        self._rollout_chunk_count = 0
        self._rollout_ids: dict[int, str] = {}
        self._rollout_chunk_indices: dict[int, int] = {}
        self._rollout_complete: dict[int, bool] = {}
        self._wrist_histories: dict[int, list[np.ndarray]] = {}
        self._state_histories: dict[int, list[np.ndarray]] = {}
        self._action_histories: dict[int, list[np.ndarray]] = {}
        self._vae_request_ahead_enabled = False
        self._vae_request_ahead_frame_stride = 0
        self._vae_request_ahead_frames = 0
        self._vae_prefetch_executor: ThreadPoolExecutor | None = None
        self._vae_prefetch_client: websocket_client_policy.WebsocketClientPolicy | None = None
        self._vae_prefetch_futures: dict[int, list[Future[dict]]] = {}

        display = remote_uri if remote_uri is not None else f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
        self.client = self._connect()
        self._server_metadata = self.client.get_server_metadata()
        self._configure_server_metadata()
        print(f"[{self.__class__.__name__}] Connected to {display}.")

    def _configure_server_metadata(self) -> None:
        self._stateful_action_rollout = bool(self._server_metadata.get("stateful_action_rollout", False))
        self._vae_request_ahead_enabled = bool(
            self._server_metadata.get("vae_causal_prefix_request_ahead_enabled", False)
        )
        self._vae_request_ahead_frame_stride = int(
            self._server_metadata.get("vae_causal_prefix_request_ahead_frame_stride", 0)
        )
        self._vae_request_ahead_frames = int(self._server_metadata.get("vae_causal_prefix_request_ahead_frames", 0))
        if self._stateful_action_rollout:
            self._rollout_chunk_len = int(self._server_metadata["action_chunk_len"])
            self._rollout_execute_horizon = int(
                self._server_metadata.get("action_execute_horizon", self._rollout_chunk_len)
            )
            self._rollout_chunk_count = int(self._server_metadata["action_chunk_count"])
            if (
                self._rollout_chunk_len < 1
                or self._rollout_execute_horizon < 1
                or self._rollout_execute_horizon > self._rollout_chunk_len
                or self._rollout_chunk_count < 1
            ):
                raise ValueError(
                    "Cosmos3 stateful rollout metadata requires a positive execute horizon no larger than "
                    "action_chunk_len and a positive action_chunk_count."
                )
            if self.open_loop_horizon is None:
                self.open_loop_horizon = self._rollout_execute_horizon
            elif self.open_loop_horizon != self._rollout_execute_horizon:
                raise ValueError(
                    "Cosmos3 stateful rollout requires open_loop_horizon to equal the server execute horizon: "
                    f"open_loop_horizon={self.open_loop_horizon}, "
                    f"execute_horizon={self._rollout_execute_horizon}."
                )
        if self._vae_request_ahead_enabled:
            if not self._stateful_action_rollout:
                raise ValueError("Cosmos3 VAE request-ahead requires stateful action rollout metadata.")
            if self._vae_request_ahead_frame_stride < 1 or self._vae_request_ahead_frames != 3:
                raise ValueError(
                    "Cosmos3 VAE request-ahead requires a positive frame stride and exactly three staged frames."
                )
            final_frame_offset = self._vae_request_ahead_frame_stride * (self._vae_request_ahead_frames + 1)
            if final_frame_offset != self._rollout_execute_horizon:
                raise ValueError(
                    "Cosmos3 VAE request-ahead must end exactly on the next action boundary: "
                    f"final_offset={final_frame_offset}, execute_horizon={self._rollout_execute_horizon}."
                )
        if "image_height" in self._server_metadata or "image_width" in self._server_metadata:
            if "image_height" not in self._server_metadata or "image_width" not in self._server_metadata:
                raise ValueError("Cosmos3 metadata must provide both image_height and image_width.")
            self._image_h = int(self._server_metadata["image_height"])
            self._image_w = int(self._server_metadata["image_width"])
        raw_keys = self._server_metadata.get("required_camera_keys")
        if raw_keys is None:
            self._camera_request_keys = None
            return
        if not isinstance(raw_keys, (list, tuple)):
            raise TypeError("Cosmos3 required_camera_keys metadata must be a list.")
        selected = []
        for raw_key in raw_keys:
            key = str(raw_key)
            if key not in self._SUPPORTED_CAMERA_KEYS:
                raise ValueError(
                    f"Cosmos3 server requested unsupported camera key {key!r}; "
                    f"supported={sorted(self._SUPPORTED_CAMERA_KEYS)}."
                )
            if key not in selected:
                selected.append(key)
        if not selected:
            raise ValueError("Cosmos3 required_camera_keys must not be empty.")
        self._camera_request_keys = tuple(selected)

    def _connect(self) -> websocket_client_policy.WebsocketClientPolicy:
        if self._remote_uri is not None:
            return websocket_client_policy.WebsocketClientPolicy(self._remote_uri)
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__,
                    e,
                    attempt + 1,
                    max_retries,
                )
                self.client = self._connect()
                self._server_metadata = self.client.get_server_metadata()
                self._configure_server_metadata()
                self._chunks.clear()
                self._counters.clear()
                self._clear_rollout_state()

    def _vae_prefetch_infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """Use a dedicated WebSocket so prefetch never races the action connection."""

        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                if self._vae_prefetch_client is None:
                    self._vae_prefetch_client = self._connect()
                return self._vae_prefetch_client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as error:
                self._vae_prefetch_client = None
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] VAE prefetch connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__,
                    error,
                    attempt + 1,
                    max_retries,
                )

    def _schedule_vae_prefetch(self, env_id: int, extracted_obs: dict) -> None:
        if not self._vae_request_ahead_enabled or self._rollout_complete[env_id]:
            return
        target_chunk = self._rollout_chunk_indices[env_id]
        if target_chunk < 1:
            return
        history_index = len(self._wrist_histories[env_id]) - 1
        base_index = (target_chunk - 1) * self._rollout_execute_horizon
        offset = history_index - base_index
        if offset <= 0 or offset % self._vae_request_ahead_frame_stride:
            return
        phase = offset // self._vae_request_ahead_frame_stride
        if not 1 <= phase <= self._vae_request_ahead_frames:
            return
        if self._vae_prefetch_executor is None:
            self._vae_prefetch_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="cosmos3-vae-prefetch",
            )
        request = {
            "vae_causal_prefix_prefetch": True,
            "vae_causal_prefix_prefetch_phase": phase,
            "rollout_id": self._rollout_ids[env_id],
            "rollout_chunk_index": target_chunk,
            "observation/wrist_image_prefetch": extracted_obs["wrist_image"].copy(),
        }
        future = self._vae_prefetch_executor.submit(self._vae_prefetch_infer_with_retry, request)
        self._vae_prefetch_futures.setdefault(env_id, []).append(future)

    def _join_vae_prefetch(self, env_id: int, *, record_profile: bool) -> None:
        futures = self._vae_prefetch_futures.pop(env_id, [])
        if not futures:
            return
        started = time.perf_counter()
        for future in futures:
            try:
                future.result()
            except Exception:
                logger.exception(
                    "[%s] VAE request-ahead failed; the action request will use the exact full VAE fallback.",
                    self.__class__.__name__,
                )
        if record_profile:
            self._record_profile("vae_prefetch_join_latency", time.perf_counter() - started)

    def _clear_rollout_state(self, env_id: int | None = None) -> None:
        stores = (
            self._rollout_ids,
            self._rollout_chunk_indices,
            self._rollout_complete,
            self._wrist_histories,
            self._state_histories,
            self._action_histories,
        )
        if env_id is None:
            for store in stores:
                store.clear()
            self._vae_prefetch_futures.clear()
            return
        for store in stores:
            store.pop(env_id, None)
        self._vae_prefetch_futures.pop(env_id, None)

    def _release_rollouts(self, rollout_ids: list[str], *, episode_reset: bool = False) -> None:
        if rollout_ids:
            request: dict[str, object] = {"reset_rollout_ids": rollout_ids}
            if episode_reset:
                request["episode_reset"] = True
            self._query_server(request)

    def _start_rollout(self, env_id: int, extracted_obs: dict) -> None:
        self._rollout_ids[env_id] = f"robolab-{uuid.uuid4().hex}"
        self._rollout_chunk_indices[env_id] = 0
        self._rollout_complete[env_id] = False
        self._wrist_histories[env_id] = []
        self._state_histories[env_id] = []
        self._action_histories[env_id] = []
        self._append_rollout_observation(env_id, extracted_obs)

    def _append_rollout_observation(self, env_id: int, extracted_obs: dict) -> None:
        state = np.concatenate(
            (extracted_obs["joint_position"], extracted_obs["gripper_position"]),
        ).astype(np.float32)
        self._wrist_histories[env_id].append(extracted_obs["wrist_image"].copy())
        self._state_histories[env_id].append(state)

    def _pack_stateful_request(self, extracted_obs: dict, instruction: str, env_id: int) -> dict:
        request = self._pack_request(extracted_obs, instruction)
        chunk_index = self._rollout_chunk_indices[env_id]
        expected_history_frames = chunk_index * self._rollout_execute_horizon + 1
        wrist_history = self._wrist_histories[env_id]
        state_history = self._state_histories[env_id]
        action_history = self._action_histories[env_id]
        if len(wrist_history) != expected_history_frames or len(state_history) != expected_history_frames:
            raise ValueError(
                "Cosmos3 stateful observation history length is inconsistent with the chunk index: "
                f"chunk={chunk_index}, wrist={len(wrist_history)}, state={len(state_history)}, "
                f"expected={expected_history_frames}."
            )
        expected_action_frames = chunk_index * self._rollout_execute_horizon
        if len(action_history) != expected_action_frames:
            raise ValueError(
                "Cosmos3 stateful action history length is inconsistent with the chunk index: "
                f"chunk={chunk_index}, action={len(action_history)}, expected={expected_action_frames}."
            )
        if action_history:
            action_history_array = np.stack(action_history).astype(np.float32)
        else:
            action_history_array = np.empty((0, state_history[-1].shape[0]), dtype=np.float32)
        request.update(
            {
                "rollout_id": self._rollout_ids[env_id],
                "rollout_chunk_index": chunk_index,
                "observation/wrist_image_history": np.stack(wrist_history),
                "observation/state_history": np.stack(state_history),
                "observation/action_history": action_history_array,
            }
        )
        return request

    def infer(self, obs: dict, instruction: str, *, env_id: int = 0) -> dict:
        if not self._stateful_action_rollout:
            return super().infer(obs, instruction, env_id=env_id)

        extracted = self._extract_observation(obs, env_id=env_id)
        if env_id not in self._rollout_ids:
            self._start_rollout(env_id, extracted)
        elif self._needs_refresh(env_id) and self._rollout_complete[env_id]:
            self._join_vae_prefetch(env_id, record_profile=False)
            self._release_rollouts([self._rollout_ids[env_id]])
            self._clear_rollout_state(env_id)
            super().reset(env_id=env_id)
            self._start_rollout(env_id, extracted)
        else:
            self._append_rollout_observation(env_id, extracted)
            self._schedule_vae_prefetch(env_id, extracted)

        if self._needs_refresh(env_id):
            if self._rollout_chunk_indices[env_id] > 0:
                self._join_vae_prefetch(env_id, record_profile=True)
            request = self._pack_stateful_request(extracted, instruction, env_id)
            response = self._query_server(request)
            chunk_index = self._rollout_chunk_indices[env_id]
            if int(response["rollout_chunk_index"]) != chunk_index:
                raise ValueError(
                    "Cosmos3 server returned the wrong rollout chunk: "
                    f"requested={chunk_index}, returned={response['rollout_chunk_index']}."
                )
            chunk = self._postprocess_chunk(self._unpack_response(response))
            if len(chunk) != self._rollout_chunk_len:
                raise ValueError(
                    f"Cosmos3 stateful server returned {len(chunk)} actions, expected {self._rollout_chunk_len}."
                )
            self._set_chunk(env_id, chunk)
            self._rollout_chunk_indices[env_id] += 1
            self._rollout_complete[env_id] = bool(response["rollout_complete"])

        action = self._next_action(env_id)
        self._action_histories[env_id].append(action.copy())
        return {"action": action, "viz": self._build_visualization(extracted)}

    def reset(self, *, env_id: int | None = None) -> None:
        if self._stateful_action_rollout:
            if env_id is None:
                rollout_ids = list(self._rollout_ids.values())
            else:
                rollout_id = self._rollout_ids.get(env_id)
                rollout_ids = [] if rollout_id is None else [rollout_id]
            env_ids = list(self._rollout_ids) if env_id is None else [env_id]
            for active_env_id in env_ids:
                self._join_vae_prefetch(active_env_id, record_profile=False)
            self._release_rollouts(rollout_ids, episode_reset=True)
            self._clear_rollout_state(env_id)
        super().reset(env_id=env_id)

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        required_image_keys = ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
        missing_image_keys = [key for key in required_image_keys if key not in image_obs]
        if missing_image_keys:
            raise KeyError(
                "Cosmos3 requires DROID left/right/wrist cameras; "
                f"missing={missing_image_keys}, available={list(image_obs)}."
            )

        image_tensors = [image_obs[key][env_id].detach() for key in required_image_keys]
        for image in image_tensors:
            if image.ndim != 3 or int(image.shape[-1]) != 3:
                raise ValueError(f"Cosmos3 camera image must be [H,W,3], got {tuple(image.shape)}.")
        batched_images = torch.stack(image_tensors, dim=0)
        if batched_images.dtype != torch.uint8:
            batched_images = batched_images.clamp(0, 255).to(torch.uint8)
        image_arrays = batched_images.contiguous().cpu().numpy()
        resized_images = [image_tools.resize_with_pad(image, self._image_h, self._image_w) for image in image_arrays]

        robot_state = raw_obs["proprio_obs"]
        joint_tensor = robot_state["arm_joint_pos"][env_id].detach().reshape(-1)
        gripper_tensor = robot_state["gripper_pos"][env_id].detach().reshape(-1)
        if joint_tensor.numel() != 7 or gripper_tensor.numel() != 1:
            raise ValueError(
                f"Cosmos3 state must be joint[7]+gripper[1], got {joint_tensor.numel()}/{gripper_tensor.numel()}."
            )
        state = torch.cat([joint_tensor, gripper_tensor], dim=0).float().cpu().numpy().astype(np.float32)

        return {
            "left_image": resized_images[0],
            "right_image": resized_images[1],
            "wrist_image": resized_images[2],
            "joint_position": state[:7],
            "gripper_position": state[7:8],
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = {
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }
        views = {wire_key: extracted_obs[source_key] for wire_key, source_key in self._SUPPORTED_CAMERA_KEYS.items()}
        if self._camera_request_keys is not None:
            request.update({key: views[key] for key in self._camera_request_keys})
            return request

        # Compatibility with older servers that publish no camera metadata.
        request.update(views)
        request["observation/image"] = self._compose_observation_image_from_views(
            wrist=extracted_obs["wrist_image"],
            left=extracted_obs["left_image"],
            right=extracted_obs["right_image"],
        )
        return request

    def _query_server(self, request: dict) -> dict:
        started = time.perf_counter()
        response = self._infer_with_retry(request)
        self._record_profile("inference_latency", time.perf_counter() - started)
        self._record_server_timing(response)
        return response

    def consume_profile_sections(self) -> list[tuple[str, float]]:
        """Return and clear timings produced by completed WebSocket refreshes."""

        sections = self._profile_sections
        self._profile_sections = []
        return sections

    def _record_profile(self, name: str, elapsed: float) -> None:
        self._profile_sections.append((name, float(elapsed)))

    def _record_server_timing(self, response: dict) -> None:
        timing = response.get("server_timing")
        if not isinstance(timing, dict):
            return
        for wire_key, section_name in (
            ("infer_ms", "server_infer_latency"),
            ("prev_total_ms", "server_prev_total_latency"),
        ):
            try:
                elapsed_ms = float(timing[wire_key])
            except (KeyError, TypeError, ValueError):
                continue
            if elapsed_ms >= 0.0 and math.isfinite(elapsed_ms):
                self._record_profile(section_name, elapsed_ms / 1000.0)

    def _unpack_response(self, response: dict) -> np.ndarray:
        if "action" not in response:
            raise KeyError(f"Cosmos3 server response must contain 'action', got keys={list(response)}.")
        actions = np.asarray(response["action"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"Cosmos3 server must return action chunk [T,D], got shape={actions.shape}.")
        return actions

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        return np.concatenate(
            (extracted_obs["left_image"], extracted_obs["wrist_image"], extracted_obs["right_image"]),
            axis=1,
        )

    def _compose_observation_image(self, extracted_obs: dict) -> np.ndarray:
        return self._compose_observation_image_from_views(
            wrist=extracted_obs["wrist_image"],
            left=extracted_obs["left_image"],
            right=extracted_obs["right_image"],
        )

    def _compose_observation_image_from_views(
        self,
        *,
        wrist: np.ndarray,
        left: np.ndarray,
        right: np.ndarray,
    ) -> np.ndarray:
        bottom_size = (self._image_h // 2, self._image_w // 2)
        left = self._resize_no_pad(left, bottom_size)
        right = self._resize_no_pad(right, bottom_size)
        return np.concatenate((wrist, np.concatenate((left, right), axis=1)), axis=0)

    @staticmethod
    def _resize_no_pad(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=size, mode="bilinear")
        return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)
