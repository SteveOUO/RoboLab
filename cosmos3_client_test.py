# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import importlib.util
import sys
import types
from concurrent.futures import Future
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

    def reset(self, *, env_id=None):
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)

    def _needs_refresh(self, env_id):
        if env_id not in self._chunks:
            return True
        horizon = len(self._chunks[env_id]) if self.open_loop_horizon is None else self.open_loop_horizon
        return self._counters[env_id] >= horizon

    def _set_chunk(self, env_id, chunk):
        self._chunks[env_id] = chunk
        self._counters[env_id] = 0

    def _next_action(self, env_id):
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action


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
    client._stateful_action_rollout = False
    client._rollout_chunk_len = 0
    client._rollout_execute_horizon = 0
    client._rollout_chunk_count = 0
    client._rollout_ids = {}
    client._rollout_chunk_indices = {}
    client._rollout_complete = {}
    client._wrist_histories = {}
    client._state_histories = {}
    client._action_histories = {}
    client._vae_request_ahead_enabled = False
    client._vae_request_ahead_frame_stride = 0
    client._vae_request_ahead_frames = 0
    client._vae_prefetch_executor = None
    client._vae_prefetch_client = None
    client._vae_prefetch_futures = {}
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


def test_stateful_rollout_sends_real_history_at_each_chunk_boundary() -> None:
    client = _client(
        {
            "image_height": 4,
            "image_width": 6,
            "required_camera_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            ],
            "stateful_action_rollout": True,
            "action_chunk_len": 2,
            "action_chunk_count": 2,
        }
    )
    requests = []

    def query(request):
        requests.append(request)
        chunk_index = int(request["rollout_chunk_index"])
        actions = np.full((2, 8), float(chunk_index), dtype=np.float32)
        actions[:, -1] = 0.75
        return {
            "action": actions,
            "rollout_chunk_index": chunk_index,
            "rollout_complete": chunk_index == 1,
        }

    client._query_server = query
    raw_obs = _raw_observation()
    for step in range(3):
        raw_obs["proprio_obs"]["arm_joint_pos"][0, 0] = float(step)
        client.infer(raw_obs, "pick banana")

    assert len(requests) == 2
    first, second = requests
    assert first["rollout_chunk_index"] == 0
    assert first["observation/wrist_image_history"].shape == (1, 4, 6, 3)
    assert first["observation/state_history"].shape == (1, 8)
    assert first["observation/action_history"].shape == (0, 8)
    assert second["rollout_id"] == first["rollout_id"]
    assert second["rollout_chunk_index"] == 1
    assert second["observation/wrist_image_history"].shape == (3, 4, 6, 3)
    np.testing.assert_array_equal(second["observation/state_history"][:, 0], np.arange(3, dtype=np.float32))
    assert second["observation/action_history"].shape == (2, 8)
    np.testing.assert_array_equal(second["observation/action_history"][:, -1], np.ones(2, dtype=np.float32))


def test_stateful_rollout_predicts_four_and_replans_after_two_actions() -> None:
    client = _client(
        {
            "image_height": 4,
            "image_width": 6,
            "required_camera_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            ],
            "stateful_action_rollout": True,
            "action_chunk_len": 4,
            "action_execute_horizon": 2,
            "action_chunk_count": 3,
        }
    )
    requests = []
    reset_requests = []

    def query(request):
        if "reset_rollout_ids" in request:
            reset_requests.append(request)
            return {"released_rollouts": len(request["reset_rollout_ids"])}
        requests.append(request)
        chunk_index = int(request["rollout_chunk_index"])
        return {
            "action": np.full((4, 8), float(chunk_index), dtype=np.float32),
            "rollout_chunk_index": chunk_index,
            "rollout_complete": chunk_index == 2,
        }

    client._query_server = query
    raw_obs = _raw_observation()
    returned = []
    for step in range(7):
        raw_obs["proprio_obs"]["arm_joint_pos"][0, 0] = float(step)
        returned.append(client.infer(raw_obs, "pick banana")["action"][0])

    assert client.open_loop_horizon == 2
    assert returned == [0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0]
    assert len(requests) == 4
    assert len(reset_requests) == 1
    assert "episode_reset" not in reset_requests[0]
    assert requests[3]["rollout_id"] != requests[0]["rollout_id"]
    assert requests[3]["rollout_chunk_index"] == 0
    assert requests[3]["observation/wrist_image_history"].shape == (1, 4, 6, 3)
    assert requests[3]["observation/action_history"].shape == (0, 8)
    assert requests[1]["observation/wrist_image_history"].shape == (3, 4, 6, 3)
    assert requests[1]["observation/action_history"].shape == (2, 8)
    assert requests[2]["observation/wrist_image_history"].shape == (5, 4, 6, 3)
    assert requests[2]["observation/action_history"].shape == (4, 8)


def test_rollout_release_marks_only_true_environment_episode_resets() -> None:
    client = _client({})
    requests: list[dict] = []
    client._query_server = lambda request: requests.append(request) or {"released_rollouts": 1}

    client._release_rollouts(["rollout-0"])
    client._release_rollouts(["rollout-1"], episode_reset=True)

    assert requests == [
        {"reset_rollout_ids": ["rollout-0"]},
        {"reset_rollout_ids": ["rollout-1"], "episode_reset": True},
    ]


class _InlineExecutor:
    def submit(self, fn, *args, **kwargs):
        future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


def test_stateful_rollout_prefetches_sampled_frames_before_next_action_request() -> None:
    client = _client(
        {
            "image_height": 4,
            "image_width": 6,
            "required_camera_keys": [
                "observation/exterior_image_1_left",
                "observation/wrist_image_left",
            ],
            "stateful_action_rollout": True,
            "action_chunk_len": 32,
            "action_execute_horizon": 8,
            "action_chunk_count": 3,
            "vae_causal_prefix_request_ahead_enabled": True,
            "vae_causal_prefix_request_ahead_frame_stride": 2,
            "vae_causal_prefix_request_ahead_frames": 3,
        }
    )
    client._vae_prefetch_executor = _InlineExecutor()
    prefetch_requests = []
    client._vae_prefetch_infer_with_retry = lambda request: (
        prefetch_requests.append(request)
        or {
            "vae_causal_prefix_prefetch": True,
            "prefetch_phase": request["vae_causal_prefix_prefetch_phase"],
        }
    )
    action_requests = []

    def query(request):
        action_requests.append(request)
        chunk_index = int(request["rollout_chunk_index"])
        return {
            "action": np.zeros((32, 8), dtype=np.float32),
            "rollout_chunk_index": chunk_index,
            "rollout_complete": False,
        }

    client._query_server = query
    raw_obs = _raw_observation()
    for step in range(9):
        raw_obs["image_obs"]["wrist_cam"].fill_(step)
        client.infer(raw_obs, "pick banana")

    assert len(action_requests) == 2
    assert action_requests[1]["rollout_chunk_index"] == 1
    assert action_requests[1]["observation/wrist_image_history"].shape == (9, 4, 6, 3)
    assert [request["vae_causal_prefix_prefetch_phase"] for request in prefetch_requests] == [1, 2, 3]
    assert [request["rollout_chunk_index"] for request in prefetch_requests] == [1, 1, 1]
    assert [int(request["observation/wrist_image_prefetch"][0, 0, 0]) for request in prefetch_requests] == [2, 4, 6]
    assert any(name == "vae_prefetch_join_latency" for name, _elapsed in client.consume_profile_sections())


def test_request_ahead_metadata_requires_exact_action_boundary() -> None:
    with pytest.raises(ValueError, match="end exactly"):
        _client(
            {
                "stateful_action_rollout": True,
                "action_chunk_len": 32,
                "action_execute_horizon": 8,
                "action_chunk_count": 3,
                "vae_causal_prefix_request_ahead_enabled": True,
                "vae_causal_prefix_request_ahead_frame_stride": 1,
                "vae_causal_prefix_request_ahead_frames": 3,
            }
        )
