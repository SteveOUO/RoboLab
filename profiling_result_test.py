# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from __future__ import annotations

import json

import pytest

from robolab.core.logging.result_summary import (
    refresh_result_yaml_server_summary,
    summarize_server_log,
)
from robolab.eval.episode import TimingStats


def test_timing_stats_exports_complete_inference_latency_distribution() -> None:
    timer = TimingStats()
    timer.add("policy_inference", 0.7)
    timer.add("env_step", 0.3)
    for elapsed in (0.1, 0.2, 0.4):
        timer.add("inference_latency", elapsed)
    timer.add("server_infer_latency", 0.65)
    timer.add("server_prev_total_latency", 0.63)

    timing = timer.to_dict(num_steps=3)

    assert timing["inference_latency_count"] == 3
    assert timing["inference_latency_total_s"] == pytest.approx(0.7)
    assert timing["inference_latency_avg_ms"] == pytest.approx(233.333)
    assert timing["inference_latency_min_ms"] == pytest.approx(100.0)
    assert timing["inference_latency_p50_ms"] == pytest.approx(200.0)
    assert timing["inference_latency_p95_ms"] == pytest.approx(380.0)
    assert timing["inference_latency_max_ms"] == pytest.approx(400.0)
    assert timing["inference_latency_samples_ms"] == pytest.approx([100.0, 200.0, 400.0])
    # Nested profile sections must not double count policy inference in wall time.
    assert timing["wall_total_s"] == pytest.approx(1.0)


def _profile_line(**payload) -> str:
    return f"[1,0]<stdout>:\x1b[32m00:00\x1b[0m [four-gpu-profile] {json.dumps(payload)}\n"


def test_four_gpu_profile_json_populates_stage_and_branch_components(tmp_path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "".join(
            [
                _profile_line(scope="coordinator_stage", request_id=-1, stage="head", total=99.0),
                _profile_line(scope="actor_stage", request_id=-1, stage="head", branch="vision_cond", total=88.0),
                _profile_line(scope="coordinator_stage", request_id=0, stage="head", total=2.0),
                _profile_line(scope="coordinator_stage", request_id=0, stage="wrist", total=0.5),
                _profile_line(scope="coordinator_stage", request_id=1, stage="wrist", total=0.7),
                _profile_line(scope="actor_stage", request_id=0, stage="head", branch="vision_cond", total=1.5),
                _profile_line(scope="actor_stage", request_id=0, stage="wrist", branch="vision_cond", total=0.4),
                _profile_line(scope="actor_stage", request_id=1, stage="wrist", branch="vision_cond", total=0.6),
                _profile_line(scope="actor_stage", request_id=0, stage="head", branch="action_cond", total=1.0),
                "[four-gpu-profile] not-json\n",
            ]
        ),
        encoding="utf-8",
    )

    summary = summarize_server_log(log_path)

    assert summary is not None
    assert summary["server_time_s"] == pytest.approx(3.2)
    components = summary["components"]
    assert components["coordinator_head"] == {
        "count": 1,
        "total_s": 2.0,
        "avg_s": 2.0,
        "p50_s": 2.0,
        "p95_s": 2.0,
        "max_s": 2.0,
    }
    assert components["coordinator_wrist"] == {
        "count": 2,
        "total_s": 1.2,
        "avg_s": 0.6,
        "p50_s": 0.6,
        "p95_s": 0.69,
        "max_s": 0.7,
    }
    assert components["actor_vision_cond_head"]["count"] == 1
    assert components["actor_vision_cond_wrist"] == {
        "count": 2,
        "total_s": 1.0,
        "avg_s": 0.5,
        "p50_s": 0.5,
        "p95_s": 0.59,
        "max_s": 0.6,
    }
    assert components["actor_action_cond_head"]["total_s"] == pytest.approx(1.0)
    assert all(component["max_s"] < 80.0 for component in components.values())


def test_refresh_result_yaml_reparses_complete_server_log_atomically(tmp_path) -> None:
    yaml = pytest.importorskip("yaml")
    log_path = tmp_path / "server.log"
    log_path.write_text(
        _profile_line(scope="coordinator_stage", request_id=0, stage="head", total=2.0)
        + _profile_line(scope="coordinator_stage", request_id=1, stage="wrist", total=0.5),
        encoding="utf-8",
    )
    result_path = tmp_path / "result.yaml"
    result_path.write_text(
        yaml.safe_dump(
            {
                "totals": {"steps": 750},
                "run": {"server_log": str(log_path)},
                "server": {"log_path": str(log_path), "components": {}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    refreshed_path = refresh_result_yaml_server_summary(result_path)

    refreshed = yaml.safe_load(refreshed_path.read_text(encoding="utf-8"))
    assert refreshed["totals"] == {"steps": 750}
    assert refreshed["server"]["components"]["coordinator_head"]["count"] == 1
    assert refreshed["server"]["components"]["coordinator_wrist"]["count"] == 1
    assert refreshed["server"]["server_time_s"] == pytest.approx(2.5)
    assert not (tmp_path / ".result.yaml.tmp").exists()
