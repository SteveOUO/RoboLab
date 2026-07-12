# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import math
import time

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

        display = remote_uri if remote_uri is not None else f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
        self.client = self._connect()
        self._server_metadata = self.client.get_server_metadata()
        self._configure_server_metadata()
        print(f"[{self.__class__.__name__}] Connected to {display}.")

    def _configure_server_metadata(self) -> None:
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
