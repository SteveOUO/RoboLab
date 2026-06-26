#!/usr/bin/env python3
"""Watch success-render manifests and convert completed RoboLab outputs to LeRobot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _entries(manifest_path: Path) -> list[dict]:
    data = json.loads(manifest_path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("entries", "items", "manifest"):
            value = data.get(key)
            if isinstance(value, list):
                return value
    raise ValueError(f"Invalid manifest format: {manifest_path}")


def _count_complete(root: Path) -> tuple[int, int, int]:
    entries = _entries(root / "render_success_manifest_threecam.json")
    complete = 0
    partial = 0
    for entry in entries:
        videos = entry["videos"]
        ready = sum(Path(path).exists() and Path(path).stat().st_size > 0 for path in videos.values())
        complete += ready == len(videos)
        partial += 0 < ready < len(videos)
    return complete, len(entries), partial


def _all_complete(roots: list[Path]) -> bool:
    ok = True
    for root in roots:
        complete, total, partial = _count_complete(root)
        print(f"[watch] {root.name}: complete={complete}/{total} partial={partial}", flush=True)
        ok = ok and complete == total and partial == 0
    return ok


def _run_convert(root: Path, fps: float, robot_type: str) -> None:
    out = root / "lerobot"
    cmd = [
        "uv", "run", "python", "scripts/convert_to_lerobot.py",
        "--input", str(root),
        "--output", str(out),
        "--robot-type", robot_type,
        "--fps", str(fps),
        "--success-only",
    ]
    print(f"[convert] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    print(f"[convert] done: {out}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--robot-type", default="droid")
    args = parser.parse_args()

    roots = [root.resolve() for root in args.roots]
    for root in roots:
        manifest = root / "render_success_manifest_threecam.json"
        if not manifest.exists():
            raise FileNotFoundError(manifest)

    while True:
        if _all_complete(roots):
            break
        time.sleep(args.poll_seconds)

    for root in roots:
        _run_convert(root, fps=args.fps, robot_type=args.robot_type)
    print("[watch] all conversions complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
