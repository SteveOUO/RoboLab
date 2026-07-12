# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from openpi_client import image_tools


class _InferenceClient:
    open_loop_horizon = None

    def __init__(self) -> None:
        self._chunks = {}
        self._counters = {}

    def infer(self, obs, instruction, *, env_id=0):
        extracted = self._extract_observation(obs, env_id=env_id)
        if env_id not in self._chunks or self._counters[env_id] >= len(self._chunks[env_id]):
            response = self._query_server(self._pack_request(extracted, instruction))
            self._chunks[env_id] = self._postprocess_chunk(self._unpack_response(response))
            self._counters[env_id] = 0
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return {"action": action, "viz": self._build_visualization(extracted)}


repo_root = Path(__file__).parent
base_client_module = types.ModuleType("robolab.eval.base_client")
base_client_module.InferenceClient = _InferenceClient
robolab_module = sys.modules.setdefault("robolab", types.ModuleType("robolab"))
robolab_module.__path__ = [str(repo_root / "robolab")]
robolab_eval_module = sys.modules.setdefault("robolab.eval", types.ModuleType("robolab.eval"))
robolab_eval_module.__path__ = [str(repo_root / "robolab" / "eval")]
sys.modules["robolab.eval.base_client"] = base_client_module
module_path = repo_root / "robolab_policy_client" / "cosmos3.py"
spec = importlib.util.spec_from_file_location("_cosmos3_client_under_test", module_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Unable to load {module_path}.")
cosmos3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cosmos3_module)
Cosmos3Client = cosmos3_module.Cosmos3Client


def _client(metadata: dict) -> Cosmos3Client:
    client = Cosmos3Client.__new__(Cosmos3Client)
    _InferenceClient.__init__(client)
    client._profile_sections = []
    client._image_h = 4
    client._image_w = 6
    client._server_metadata = metadata
    client._configure_server_metadata()
    return client


def _raw_observation() -> dict:
    base = torch.arange(3 * 5 * 3, dtype=torch.uint8).reshape(3, 5, 3)
    return {
        "image_obs": {
            "over_shoulder_left_camera": base.unsqueeze(0),
            "over_shoulder_right_camera": (base + 1).unsqueeze(0),
            "wrist_cam": (base + 2).unsqueeze(0),
        },
        "proprio_obs": {
            "arm_joint_pos": torch.arange(7, dtype=torch.float32).unsqueeze(0),
            "gripper_pos": torch.tensor([[0.25]], dtype=torch.float32),
        },
    }


def test_metadata_slims_wire_request_to_required_views() -> None:
    client = _client(
        {
            "image_height": 4,
            "image_width": 6,
            "required_camera_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            ],
        }
    )
    extracted = client._extract_observation(_raw_observation())

    request = client._pack_request(extracted, "pick banana")

    assert set(request) == {
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    }
    np.testing.assert_array_equal(extracted["joint_position"], np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(extracted["gripper_position"], np.array([0.25], dtype=np.float32))


def test_batched_extract_preserves_existing_resize_with_pad_pixels() -> None:
    client = _client({})
    raw = _raw_observation()

    extracted = client._extract_observation(raw)

    for source_key, extracted_key in (
        ("over_shoulder_left_camera", "left_image"),
        ("over_shoulder_right_camera", "right_image"),
        ("wrist_cam", "wrist_image"),
    ):
        source = raw["image_obs"][source_key][0].numpy()
        expected = image_tools.resize_with_pad(source, 4, 6)
        np.testing.assert_array_equal(extracted[extracted_key], expected)


def test_missing_metadata_retains_legacy_wire_contract() -> None:
    client = _client({})
    extracted = client._extract_observation(_raw_observation())

    request = client._pack_request(extracted, "pick banana")

    assert {
        "observation/image",
        "observation/exterior_image_1_left",
        "observation/exterior_image_2_left",
        "observation/wrist_image_left",
    }.issubset(request)


def test_metadata_rejects_unsupported_camera_key() -> None:
    with pytest.raises(ValueError, match="unsupported camera key"):
        _client({"required_camera_keys": ["observation/unknown"]})


def test_profile_sections_only_record_completed_websocket_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _client({})
    client._extract_observation = lambda _obs, env_id=0: {}
    client._pack_request = lambda _obs, _instruction: {}
    client._build_visualization = lambda _obs: None
    responses = iter(
        [
            {
                "action": np.zeros((2, 8), dtype=np.float32),
                "server_timing": {"infer_ms": 80.0, "prev_total_ms": 90.0},
            },
            {
                "action": np.ones((2, 8), dtype=np.float32),
                "server_timing": {"infer_ms": 120.0},
            },
        ]
    )
    client._infer_with_retry = lambda _request: next(responses)
    clock = iter((10.0, 10.1, 20.0, 20.2))
    monkeypatch.setattr(cosmos3_module.time, "perf_counter", lambda: next(clock))

    client.infer({}, "pick banana")
    assert client.consume_profile_sections() == [
        ("inference_latency", pytest.approx(0.1)),
        ("server_infer_latency", pytest.approx(0.08)),
        ("server_prev_total_latency", pytest.approx(0.09)),
    ]

    client.infer({}, "pick banana")
    assert client.consume_profile_sections() == []

    client.infer({}, "pick banana")
    assert client.consume_profile_sections() == [
        ("inference_latency", pytest.approx(0.2)),
        ("server_infer_latency", pytest.approx(0.12)),
    ]
