#!/usr/bin/env python3
"""Finalize a partially written RoboLab LeRobot v3 export.

This is useful when the expensive video conversion finished but the process
stopped before writing meta/stats.json or meta/info.json.
"""

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def _to_float_list(value):
    arr = np.asarray(value)
    return arr.tolist() if arr.shape != () else [float(arr)]


def _feature_names(key: str, length: int):
    if "joint" in key.lower() or key == "observation.state":
        return {"motors": [f"joint_{i}" for i in range(length)]}
    if key == "action":
        if length == 8:
            return {"motors": [f"joint_{i}" for i in range(7)] + ["gripper"]}
        return {"motors": [f"action_{i}" for i in range(length)]}
    if "position" in key.lower() and length == 3:
        return ["x", "y", "z"]
    if "position" in key.lower() and length == 7:
        return ["x", "y", "z", "qx", "qy", "qz", "qw"]
    if "orientation" in key.lower() and length == 4:
        return ["qx", "qy", "qz", "qw"]
    return None


def _probe_video(path: Path, fallback_fps: int):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        streams = json.loads(result.stdout).get("streams") or []
        if streams:
            stream = streams[0]
            width = int(stream.get("width") or 0)
            height = int(stream.get("height") or 0)
            rate = stream.get("r_frame_rate") or "0/1"
            try:
                fps = float(Fraction(rate))
            except (ValueError, ZeroDivisionError):
                fps = float(fallback_fps)
            return width, height, fps if fps > 0 else float(fallback_fps)
    except Exception:
        pass
    return 0, 0, float(fallback_fps)


def _video_features(dataset_dir: Path, fps: int):
    features = {}
    videos_root = dataset_dir / "videos"
    if not videos_root.exists():
        return features
    for camera_dir in sorted(p for p in videos_root.iterdir() if p.is_dir()):
        candidates = sorted(camera_dir.glob("chunk-*/*.mp4"))
        if not candidates:
            continue
        width, height, video_fps = _probe_video(candidates[0], fps)
        features[camera_dir.name] = {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "avc1",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": float(video_fps),
                "video.channels": 3,
                "has_audio": False,
            },
        }
    return features


def _compute_stats(table):
    stats = {}
    for field in table.schema:
        key = field.name
        if key == "next.done":
            continue
        col = table[key].to_pylist()
        if not col:
            continue
        first = next((v for v in col if v is not None), None)
        if first is None or isinstance(first, str):
            continue
        if isinstance(first, list):
            values = np.asarray(col, dtype=np.float32)
        elif isinstance(first, (int, float, bool)):
            values = np.asarray(col, dtype=np.float32).reshape(-1, 1)
        else:
            continue
        stats[key] = {
            "mean": _to_float_list(values.mean(axis=0)),
            "std": _to_float_list(np.nan_to_num(values.std(axis=0), nan=0.0, posinf=0.0, neginf=0.0)),
            "min": _to_float_list(values.min(axis=0)),
            "max": _to_float_list(values.max(axis=0)),
            "count": [int(values.shape[0])],
        }
    return stats


def _image_stats(episode_table, video_features):
    lengths = episode_table["length"].to_pylist() if "length" in episode_table.column_names else []
    total_count = int(sum(int(x) for x in lengths))
    return {
        key: {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.5]], [[0.5]], [[0.5]]],
            "count": [total_count],
        }
        for key in video_features
    }


def _features(table, video_features, fps: int):
    features = {}
    first_rows = table.slice(0, 1).to_pylist()
    first_row = first_rows[0] if first_rows else {}
    for key, value in first_row.items():
        if key in ("episode_index", "frame_index", "index", "task_index"):
            features[key] = {"dtype": "int64", "shape": [1], "names": None, "fps": fps}
        elif key == "timestamp":
            features[key] = {"dtype": "float32", "shape": [1], "names": None, "fps": fps}
        elif key == "next.done":
            features[key] = {"dtype": "bool", "shape": [1], "names": None, "fps": fps}
        elif isinstance(value, str):
            features[key] = {"dtype": "string", "shape": [1], "names": None, "fps": fps}
        elif isinstance(value, list):
            features[key] = {
                "dtype": "float32",
                "shape": [len(value)],
                "names": _feature_names(key, len(value)),
                "fps": fps,
            }
    features.update(video_features)
    return features


def finalize(dataset_dir: Path, robot_type: str, fps: int):
    data_path = dataset_dir / "data" / "chunk-000" / "file-000.parquet"
    episodes_path = dataset_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    tasks_jsonl = dataset_dir / "meta" / "tasks.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not episodes_path.exists():
        raise FileNotFoundError(episodes_path)

    data_table = pq.read_table(data_path)
    episode_table = pq.read_table(episodes_path)
    video_features = _video_features(dataset_dir, fps)

    stats = _compute_stats(data_table)
    stats.update(_image_stats(episode_table, video_features))
    stats_path = dataset_dir / "meta" / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    total_tasks = 0
    if tasks_jsonl.exists():
        total_tasks = sum(1 for line in tasks_jsonl.read_text().splitlines() if line.strip())

    info = {
        "codebase_version": "v3.0",
        "robot_type": robot_type,
        "total_episodes": int(episode_table.num_rows),
        "total_frames": int(data_table.num_rows),
        "total_tasks": int(total_tasks),
        "total_videos": len(video_features),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": int(fps),
        "splits": {"train": f"0:{int(episode_table.num_rows)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": _features(data_table, video_features, int(fps)),
        "data_files_size_in_mb": int(data_path.stat().st_size / 1024 / 1024),
        "video_files_size_in_mb": int(sum(p.stat().st_size for p in (dataset_dir / "videos").glob("**/*.mp4")) / 1024 / 1024),
    }
    info_path = dataset_dir / "meta" / "info.json"
    info_path.write_text(json.dumps(info, indent=2))
    print(f"Wrote {stats_path}")
    print(f"Wrote {info_path}")
    print(f"episodes={info['total_episodes']} frames={info['total_frames']} tasks={info['total_tasks']} videos={info['total_videos']}")


def main():
    parser = argparse.ArgumentParser(description="Finalize a RoboLab LeRobot v3 export")
    parser.add_argument("dataset_dir")
    parser.add_argument("--robot-type", default="droid")
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()
    finalize(Path(args.dataset_dir), robot_type=args.robot_type, fps=args.fps)


if __name__ == "__main__":
    main()
