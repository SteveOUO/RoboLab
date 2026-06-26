#!/usr/bin/env python3
"""Build render manifests for successful RoboLab episodes.

The generated entries are consumed by render_hdf5_state_videos.py.  Per-camera
MP4 names intentionally match LeRobotExporter's RoboLab glob pattern:
    *_{run}_env{env_id}__{camera_name}.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_CAMERAS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)


def build_manifest(root: Path, cameras: tuple[str, ...], output_name: str) -> Path:
    root = root.resolve()
    results_path = root / "episode_results.jsonl"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing episode results: {results_path}")

    entries: list[dict] = []
    skipped_missing_hdf5 = 0
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            result = json.loads(line)
            if not result.get("success", False):
                continue

            task = result["task_name"]
            run = int(result.get("run", 0))
            env_id = int(result["env_id"])
            task_dir = root / task
            hdf5_path = task_dir / f"run_{run}.hdf5"
            if not hdf5_path.exists():
                skipped_missing_hdf5 += 1
                continue

            prefix = f"success_{run}_env{env_id}"
            entries.append(
                {
                    "task": task,
                    "hdf5": str(hdf5_path),
                    "demo": f"demo_{env_id}",
                    "result": {
                        "run": run,
                        "episode": int(result.get("episode", env_id)),
                        "env_id": env_id,
                        "instruction": result.get("instruction", ""),
                        "policy": result.get("policy", ""),
                    },
                    "videos": {
                        camera: str(task_dir / f"{prefix}__{camera}.mp4")
                        for camera in cameras
                    },
                }
            )

    manifest = {
        "source": str(root),
        "success_count": len(entries),
        "skipped_missing_hdf5": skipped_missing_hdf5,
        "cameras": list(cameras),
        "entries": entries,
    }
    output_path = root / output_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output-name", default="render_success_manifest_threecam.json")
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    args = parser.parse_args()

    cameras = tuple(args.cameras)
    for root in args.roots:
        path = build_manifest(root, cameras, args.output_name)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"{path}: {data['success_count']} entries, skipped_missing_hdf5={data['skipped_missing_hdf5']}")


if __name__ == "__main__":
    main()
