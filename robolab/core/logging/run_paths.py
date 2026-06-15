# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Run directory helpers for RoboLab evaluation.

New RoboLab eval outputs are organized by UTC+8 date and minute:

  examples/Droid/RoboLab/output/YYYYMMDD/HHMM/<run_name>/
  runs/logs/YYYYMMDD/HHMM/

Existing flat output folders remain resumable when explicitly requested.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TextIO

from robolab.constants import DEFAULT_OUTPUT_DIR, PACKAGE_DIR

UTC8 = timezone(timedelta(hours=8))


def utc8_now() -> datetime:
    return datetime.now(tz=UTC8)


def utc8_stamp(now: datetime | None = None) -> tuple[str, str]:
    now = now or utc8_now()
    return now.strftime("%Y%m%d"), now.strftime("%H%M")


def utc8_timestamp(now: datetime | None = None) -> str:
    now = now or utc8_now()
    return now.strftime("%Y%m%d_%H%M%S")


def _is_valid_run_stamp(run_date: str | None, run_time: str | None) -> bool:
    return bool(
        run_date
        and run_time
        and re.fullmatch(r"\d{8}", run_date)
        and re.fullmatch(r"\d{4}", run_time)
    )


def ensure_run_stamp_env() -> tuple[str, str]:
    run_date = os.environ.get("STARVLA_RUN_DATE")
    run_time = os.environ.get("STARVLA_RUN_TIME")
    if not _is_valid_run_stamp(run_date, run_time):
        run_date, run_time = utc8_stamp()
        os.environ["STARVLA_RUN_DATE"] = run_date
        os.environ["STARVLA_RUN_TIME"] = run_time
    return run_date, run_time


def starvla_root() -> Path:
    return Path(PACKAGE_DIR).resolve().parents[2]


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return safe.strip("._") or "run"


def default_run_name(policy: str, suffix: str | None = None, instruction_type: str | None = None) -> str:
    run_date, run_time = ensure_run_stamp_env()
    timestamp = f"{run_date}_{run_time}{utc8_now().strftime('%S')}"
    parts = [timestamp, policy]
    if suffix:
        parts.append(suffix)
    if instruction_type and instruction_type != "default":
        parts.append(instruction_type)
    return "_".join(parts)


def resolve_output_dir(output_folder_name: str, *, output_root: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR) -> Path:
    """Resolve a RoboLab eval output directory.

    A single-component new run name is placed under output/YYYYMMDD/HHMM/.
    Explicit absolute paths, multi-component relative paths, and existing flat
    output folders are left as-is so old runs can still be resumed.
    """
    raw = Path(output_folder_name)
    if raw.is_absolute():
        return raw

    output_root_path = Path(output_root)
    direct = output_root_path / raw
    if direct.exists() or len(raw.parts) > 1:
        return direct

    run_date, run_time = ensure_run_stamp_env()
    return output_root_path / run_date / run_time / raw


def run_log_dir(*, create: bool = True) -> Path:
    explicit = os.environ.get("STARVLA_LOG_DIR")
    if explicit:
        path = Path(explicit)
    else:
        run_date, run_time = ensure_run_stamp_env()
        root = Path(os.environ.get("STARVLA_LOG_ROOT", starvla_root() / "runs" / "logs"))
        path = root / run_date / run_time
        os.environ.setdefault("STARVLA_LOG_DIR", str(path))
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


class TeeStream:
    def __init__(self, primary: TextIO, log_file: TextIO):
        self.primary = primary
        self.log_file = log_file

    def write(self, data: str) -> int:
        self.primary.write(data)
        self.log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.primary.isatty()

    def __getattr__(self, name: str):
        return getattr(self.primary, name)


def install_client_log(run_name: str) -> Path | None:
    """Mirror Python stdout/stderr to runs/logs/YYYYMMDD/HHMM/client_<run>.log.

    Shell launchers set STARVLA_LOG_ACTIVE when they already tee the full
    process output; in that case we only return the intended path.
    """
    log_dir = run_log_dir(create=True)
    log_path = Path(os.environ.get("STARVLA_CLIENT_LOG_PATH", log_dir / f"client_{sanitize_filename(run_name)}.log"))
    os.environ.setdefault("STARVLA_CLIENT_LOG_PATH", str(log_path))

    if os.environ.get("STARVLA_LOG_ACTIVE") or os.environ.get("STARVLA_DISABLE_PYTHON_TEE") in {"1", "true", "TRUE"}:
        return log_path

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", buffering=1)
    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)
    os.environ["STARVLA_LOG_ACTIVE"] = "python"
    print(f"[RoboLab] client log: {log_path}")
    return log_path
    return log_path

