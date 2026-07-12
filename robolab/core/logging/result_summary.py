# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Human-readable result.yaml writer for RoboLab evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from robolab.core.logging.run_paths import run_log_dir, starvla_root, utc8_now

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
FOUR_GPU_PROFILE_MARKER = "[four-gpu-profile]"
FOUR_GPU_BRANCHES = ("vision_cond", "vision_uncond", "action_cond", "action_uncond")
FOUR_GPU_STAGES = ("head", "wrist")


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _round(value: Any, digits: int = 3) -> float | None:
    v = _finite(value)
    return round(v, digits) if v is not None else None


def _sum(values: list[Any], digits: int = 3) -> float:
    total = sum(v for v in (_finite(x) for x in values) if v is not None)
    return round(total, digits)


def _mean(values: list[Any], digits: int = 3) -> float | None:
    numeric = [v for v in (_finite(x) for x in values) if v is not None]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), digits)


def _metric(ep: dict, key: str) -> Any:
    return ep.get("metrics", {}).get(key, ep.get(key))


def _timing(ep: dict, key: str) -> Any:
    return ep.get("timing", {}).get(key)


def _percentile(values: list[float], percentile: float, digits: int = 1) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return round(sorted_values[0], digits)
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    value = sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
    return round(value, digits)


def _seconds_summary(values: list[float]) -> dict[str, Any]:
    total = sum(values)
    return {
        "count": len(values),
        "total_s": round(total, 3),
        "avg_s": round(total / len(values), 3),
        "p50_s": _percentile(values, 50, 3),
        "p95_s": _percentile(values, 95, 3),
        "max_s": round(max(values), 3),
    }


def _four_gpu_profile_summary(text: str) -> tuple[dict[str, dict[str, Any]], float | None]:
    """Parse structured four-GPU stage profiles, ignoring synthetic warmup."""

    grouped: dict[tuple[str, str, str | None], list[float]] = defaultdict(list)
    decoder = json.JSONDecoder()
    for line in text.splitlines():
        _, marker, suffix = line.partition(FOUR_GPU_PROFILE_MARKER)
        if not marker:
            continue
        try:
            record, _ = decoder.raw_decode(suffix.lstrip())
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(record, dict):
            continue
        request_id = _finite(record.get("request_id"))
        if request_id is None or request_id < 0:
            continue
        stage = str(record.get("stage", ""))
        if stage not in FOUR_GPU_STAGES:
            continue
        scope = record.get("scope")
        elapsed = _finite(record.get("total"))
        if elapsed is None or elapsed < 0:
            continue
        if scope == "coordinator_stage":
            grouped[("coordinator", stage, None)].append(elapsed)
        elif scope == "actor_stage":
            branch = str(record.get("branch", ""))
            if branch in FOUR_GPU_BRANCHES:
                grouped[("actor", stage, branch)].append(elapsed)

    components: dict[str, dict[str, Any]] = {}
    coordinator_totals: list[float] = []
    for stage in FOUR_GPU_STAGES:
        values = grouped.get(("coordinator", stage, None), [])
        if values:
            components[f"coordinator_{stage}"] = _seconds_summary(values)
            coordinator_totals.extend(values)
    for branch in FOUR_GPU_BRANCHES:
        for stage in FOUR_GPU_STAGES:
            values = grouped.get(("actor", stage, branch), [])
            if values:
                components[f"actor_{branch}_{stage}"] = _seconds_summary(values)

    server_time_s = round(sum(coordinator_totals), 3) if coordinator_totals else None
    return components, server_time_s


def _inference_latency_summary(episodes: list[dict]) -> dict[str, Any]:
    samples_ms: list[float] = []
    for ep in episodes:
        raw_samples = _timing(ep, "inference_latency_samples_ms")
        if not isinstance(raw_samples, list):
            continue
        samples_ms.extend(v for v in (_finite(x) for x in raw_samples) if v is not None)

    if samples_ms:
        total_ms = sum(samples_ms)
        return {
            "count": len(samples_ms),
            "total_s": round(total_ms / 1000.0, 3),
            "avg_ms": round(total_ms / len(samples_ms), 1),
            "min_ms": round(min(samples_ms), 1),
            "p50_ms": _percentile(samples_ms, 50),
            "p95_ms": _percentile(samples_ms, 95),
            "max_ms": round(max(samples_ms), 1),
        }

    count = 0
    total_s = 0.0
    mins_ms: list[float] = []
    maxes_ms: list[float] = []
    for ep in episodes:
        ep_count = _finite(_timing(ep, "inference_latency_count"))
        if ep_count is None:
            continue
        count += int(ep_count)
        total_s += (
            _finite(_timing(ep, "inference_latency_total_s")) or _finite(_timing(ep, "inference_latency_s")) or 0.0
        )
        ep_min = _finite(_timing(ep, "inference_latency_min_ms"))
        ep_max = _finite(_timing(ep, "inference_latency_max_ms"))
        if ep_min is not None:
            mins_ms.append(ep_min)
        if ep_max is not None:
            maxes_ms.append(ep_max)

    return {
        "count": count,
        "total_s": round(total_s, 3),
        "avg_ms": round(total_s / count * 1000.0, 1) if count else None,
        "min_ms": round(min(mins_ms), 1) if mins_ms else None,
        "p50_ms": None,
        "p95_ms": None,
        "max_ms": round(max(maxes_ms), 1) if maxes_ms else None,
    }


def _success_rate(successes: int, total: int) -> float:
    return round(successes / total, 4) if total else 0.0


def _strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def _compact_log_text(path: Path) -> str:
    text = path.read_text(errors="replace")
    text = _strip_ansi(text)
    return re.sub(r"\s+", "", text)


def summarize_server_log(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not path:
        return None
    log_path = Path(path)
    if not log_path.exists():
        return {"log_path": str(log_path), "found": False}

    text = _strip_ansi(log_path.read_text(errors="replace"))
    compact = re.sub(r"\s+", "", text)
    four_gpu_components, four_gpu_server_time = _four_gpu_profile_summary(text)
    components = {
        "websocket": r"websockettiming",
        "coordinator": r"coordinatortiming",
        "action_actor": r"actionactortiming",
        "video_actor": r"videoactortiming",
        "frame_actor": r"frameactortiming",
        "cfg_actor": r"cfgactortiming",
    }
    summary: dict[str, Any] = {
        "log_path": str(log_path),
        "found": True,
        "components": dict(four_gpu_components),
    }
    output_dirs: dict[str, str] = {}
    debug_matches = re.findall(r"DROID server debug output dir:\s*(\S+)", text)
    video_matches = re.findall(r"DROID server video output dir:\s*(\S+)", text)
    if debug_matches:
        output_dirs["debug_dir"] = debug_matches[-1]
    if video_matches:
        output_dirs["video_dir"] = video_matches[-1]
    if output_dirs:
        summary["outputs"] = output_dirs
    for name, marker in components.items():
        totals = [float(m.group(1)) for m in re.finditer(marker + r".*?total=([0-9.]+)s", compact)]
        if totals:
            summary["components"][name] = {
                "count": len(totals),
                "total_s": round(sum(totals), 3),
                "avg_s": round(sum(totals) / len(totals), 3),
                "max_s": round(max(totals), 3),
            }

    # Non-Wan single-process logs use "DROID ... websocket timing".
    if not summary["components"]:
        totals = [float(m.group(1)) for m in re.finditer(r"websockettiming.*?total=([0-9.]+)s", compact)]
        infers = [float(m.group(1)) for m in re.finditer(r"infer[:=]([0-9.]+)", compact)]
        if totals:
            summary["components"]["websocket"] = {
                "count": len(totals),
                "total_s": round(sum(totals), 3),
                "avg_s": round(sum(totals) / len(totals), 3),
                "max_s": round(max(totals), 3),
            }
        if infers:
            summary["components"]["infer"] = {
                "count": len(infers),
                "total_s": round(sum(infers), 3),
                "avg_s": round(sum(infers) / len(infers), 3),
                "max_s": round(max(infers), 3),
            }

    websocket = summary["components"].get("websocket")
    summary["server_time_s"] = four_gpu_server_time
    if summary["server_time_s"] is None and websocket:
        summary["server_time_s"] = websocket.get("total_s")
    return summary


def _discover_server_log(log_dir: Path) -> str | None:
    env_path = os.environ.get("STARVLA_SERVER_LOG_PATH")
    if env_path:
        return env_path
    candidates = sorted(log_dir.glob("server*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return str(candidates[0])
    logs_root = Path(os.environ.get("STARVLA_LOG_ROOT", starvla_root() / "runs" / "logs"))
    candidates = sorted(logs_root.glob("**/server*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else None


def _run_tag(args: Any) -> str:
    return str(
        getattr(args, "run_tag", None) or os.environ.get("STARVLA_RUN_TAG") or os.environ.get("STARVLA_RUN_LABEL") or ""
    )


def build_result_summary(
    *,
    episode_results: list[dict],
    output_dir: str,
    run_name: str,
    args: Any,
    started_at: datetime,
    ended_at: datetime | None = None,
    filter_str: str | None = None,
) -> dict[str, Any]:
    ended_at = ended_at or utc8_now()
    log_dir = run_log_dir(create=True)
    server_log = _discover_server_log(log_dir)
    total = len(episode_results)
    successes = sum(1 for ep in episode_results if ep.get("success"))
    total_steps = sum(int(ep.get("episode_step") or 0) for ep in episode_results)

    tasks: dict[str, list[dict]] = defaultdict(list)
    for ep in episode_results:
        tasks[ep.get("env_name") or ep.get("task_name") or "unknown"].append(ep)

    task_rows = []
    for task_name in sorted(tasks):
        eps = tasks[task_name]
        task_successes = sum(1 for ep in eps if ep.get("success"))
        task_steps = [ep.get("episode_step") for ep in eps]
        task_rows.append(
            {
                "name": task_name,
                "task_name": eps[0].get("task_name"),
                "episodes": len(eps),
                "successes": task_successes,
                "success_rate": _success_rate(task_successes, len(eps)),
                "steps_total": sum(int(ep.get("episode_step") or 0) for ep in eps),
                "steps_avg": _mean(task_steps, 1),
                "duration_avg_s": _mean([ep.get("duration") for ep in eps], 3),
                "client_wall_total_s": _sum([_timing(ep, "wall_total_s") for ep in eps], 3),
                "policy_inference_s": _sum([_timing(ep, "policy_inference_s") for ep in eps], 3),
                "env_step_s": _sum([_timing(ep, "env_step_s") for ep in eps], 3),
                "it_per_sec_avg": _mean([_timing(ep, "it_per_sec") for ep in eps], 3),
                "inference_latency": _inference_latency_summary(eps),
                "metrics": {
                    "ee_sparc_avg": _mean([_metric(ep, "ee_sparc") for ep in eps], 3),
                    "ee_path_length_avg_m": _mean([_metric(ep, "ee_path_length") for ep in eps], 3),
                    "ee_speed_avg_cm_s": _mean(
                        [
                            (_finite(_metric(ep, "ee_speed_mean")) or 0.0) * 100
                            for ep in eps
                            if _finite(_metric(ep, "ee_speed_mean")) is not None
                        ],
                        3,
                    ),
                },
            }
        )

    client_wall = _sum([_timing(ep, "wall_total_s") for ep in episode_results], 3)
    summary = {
        "tag": _run_tag(args),
        "run": {
            "name": run_name,
            "started_at_utc8": started_at.isoformat(timespec="seconds"),
            "ended_at_utc8": ended_at.isoformat(timespec="seconds"),
            "total_runtime_s": round((ended_at - started_at).total_seconds(), 3),
            "output_dir": str(output_dir),
            "log_dir": str(log_dir),
            "client_log": os.environ.get("STARVLA_CLIENT_LOG_PATH"),
            "server_log": server_log,
        },
        "config": {
            "policy": getattr(args, "policy", None),
            "remote": {
                "host": getattr(args, "remote_host", None),
                "port": getattr(args, "remote_port", None),
                "uri": getattr(args, "remote_uri", None),
            },
            "filter": filter_str,
            "num_envs": getattr(args, "num_envs", None),
            "num_runs": getattr(args, "num_runs", None),
            "video_mode": getattr(args, "video_mode", None),
            "instruction_type": getattr(args, "instruction_type", None),
        },
        "totals": {
            "datasets": len(tasks),
            "episodes": total,
            "successes": successes,
            "success_rate": _success_rate(successes, total),
            "steps": total_steps,
            "client_time_s": {
                "wall_total": client_wall,
                "policy_inference": _sum([_timing(ep, "policy_inference_s") for ep in episode_results], 3),
                "env_step": _sum([_timing(ep, "env_step_s") for ep in episode_results], 3),
                "video_write": _sum([_timing(ep, "video_write_s") for ep in episode_results], 3),
            },
            "inference_latency": _inference_latency_summary(episode_results),
        },
        "server": summarize_server_log(server_log),
        "datasets": task_rows,
    }
    return summary


def _fallback_yaml(data: Any, indent: int = 0) -> str:
    space = " " * indent
    if isinstance(data, dict):
        lines = []
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.append(_fallback_yaml(value, indent + 2))
            else:
                lines.append(f"{space}{key}: {value}")
        return "\n".join(lines)
    if isinstance(data, list):
        lines = []
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{space}-")
                lines.append(_fallback_yaml(item, indent + 2))
            else:
                lines.append(f"{space}- {item}")
        return "\n".join(lines)
    return f"{space}{data}"


def write_result_yaml(summary: dict[str, Any], log_dir: str | os.PathLike[str] | None = None) -> Path:
    path = Path(log_dir) if log_dir else run_log_dir(create=True)
    path.mkdir(parents=True, exist_ok=True)
    result_file = path / "result.yaml"
    try:
        import yaml

        text = yaml.safe_dump(summary, sort_keys=False, allow_unicode=False)
    except Exception:
        text = _fallback_yaml(summary) + "\n"
    result_file.write_text(text)
    print(f"[RoboLab] result summary: {result_file}")
    return result_file


def refresh_result_yaml_server_summary(
    result_file: str | os.PathLike[str],
    server_log: str | os.PathLike[str] | None = None,
) -> Path:
    """Refresh only the server section after the policy server has stopped.

    MPI launchers may buffer the final stdout records beyond the point where
    the remote RoboLab client first writes ``result.yaml``.  Re-reading the
    completed server log after launcher shutdown keeps all client-side fields
    untouched while making the structured stage counts complete.
    """

    result_path = Path(result_file)
    import yaml

    summary = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError(f"result file must contain a mapping: {result_path}")

    if server_log is None:
        existing_server = summary.get("server")
        if isinstance(existing_server, dict):
            server_log = existing_server.get("log_path")
        if not server_log:
            run = summary.get("run")
            if isinstance(run, dict):
                server_log = run.get("server_log")
    if not server_log:
        raise ValueError(f"no server log path recorded in {result_path}")

    refreshed_server = summarize_server_log(server_log)
    if refreshed_server is None or not refreshed_server.get("found"):
        raise FileNotFoundError(f"server log not found: {server_log}")
    summary["server"] = refreshed_server

    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    temporary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    temporary_path.replace(result_path)
    print(f"[RoboLab] refreshed server summary: {result_path}")
    return result_path


def _main() -> None:
    parser = argparse.ArgumentParser(description="RoboLab result summary maintenance")
    parser.add_argument("--refresh-server-summary", metavar="RESULT_YAML")
    parser.add_argument("--server-log")
    args = parser.parse_args()
    if not args.refresh_server_summary:
        parser.error("--refresh-server-summary is required")
    refresh_result_yaml_server_summary(args.refresh_server_summary, args.server_log)


if __name__ == "__main__":
    _main()
