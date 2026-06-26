# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

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
    OPEN_LOOP_HORIZON = 32

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int | None = None,
        remote_uri: str | None = None,
    ) -> None:
        super().__init__()
        self._remote_host = remote_host
        self._remote_port = int(remote_port)
        self._remote_uri = remote_uri
        self.open_loop_horizon = self.OPEN_LOOP_HORIZON if open_loop_horizon is None else int(open_loop_horizon)

        self._image_w = self.IMAGE_W
        self._image_h = self.IMAGE_H

        display = remote_uri if remote_uri is not None else f"{self._remote_host}:{self._remote_port}"
        print(f"[{self.__class__.__name__}] Awaiting for server on {display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {display}.")

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

        left_image = image_obs["over_shoulder_left_camera"][env_id].clone().detach().cpu().numpy()
        right_image = image_obs["over_shoulder_right_camera"][env_id].clone().detach().cpu().numpy()
        wrist_image = image_obs["wrist_cam"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)

        return {
            "left_image": left_image,
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "observation/image": self._compose_observation_image(extracted_obs),
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

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
        left = image_tools.resize_with_pad(extracted_obs["left_image"], self._image_h, self._image_w)
        wrist = image_tools.resize_with_pad(extracted_obs["wrist_image"], self._image_h, self._image_w)
        right = image_tools.resize_with_pad(extracted_obs["right_image"], self._image_h, self._image_w)
        return np.concatenate((left, wrist, right), axis=1)

    def _compose_observation_image(self, extracted_obs: dict) -> np.ndarray:
        wrist = image_tools.resize_with_pad(extracted_obs["wrist_image"], self._image_h, self._image_w)
        left = image_tools.resize_with_pad(extracted_obs["left_image"], self._image_h, self._image_w)
        right = image_tools.resize_with_pad(extracted_obs["right_image"], self._image_h, self._image_w)

        bottom_size = (self._image_h // 2, self._image_w // 2)
        left = self._resize_no_pad(left, bottom_size)
        right = self._resize_no_pad(right, bottom_size)
        return np.concatenate((wrist, np.concatenate((left, right), axis=1)), axis=0)

    @staticmethod
    def _resize_no_pad(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=size, mode="bilinear")
        return tensor.squeeze(0).permute(1, 2, 0).numpy().astype(image.dtype)
