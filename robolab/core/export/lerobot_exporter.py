"""LeRobot v3.0 dataset exporter for RoboLab.

This module converts RoboLab's HDF5 output format to LeRobot v3.0 format,
enabling visualization in the LeRobot dataset visualizer and compatibility
with LeRobot training pipelines.

Hugging Face visualization: The export is aligned with reference datasets
(e.g. lerobot/aloha_static_cups_open) so the HF visualizer loads correctly:
- meta/info.json uses integer fps and includes data_files_size_in_mb,
  video_files_size_in_mb; features include next.done.
- Episodes parquet includes success; video metadata points only to existing
  files (file_index=-1 when an episode has no video for that camera).
- Data parquet includes next.done (true on last frame of each episode).

LeRobot v3.0 format structure:
    dataset/
    ├── meta/
    │   ├── info.json              # Dataset metadata (features, fps, etc.)
    │   ├── stats.json             # Feature statistics (mean/std/min/max)
    │   ├── tasks.jsonl            # Task descriptions
    │   └── episodes/
    │       └── chunk-000/
    │           └── file-000.parquet  # Episode metadata
    ├── data/
    │   └── chunk-000/
    │       └── file-000.parquet   # Tabular data (states, actions, timestamps)
    └── videos/
        └── <camera_name>/
            └── chunk-000/
                └── file-000.mp4   # One concatenated MP4 per camera (all episodes); episode clip via from_timestamp/to_timestamp

Videos: By default all episode videos per camera are concatenated into one MP4 (LeRobot v3
convention). Set concatenate_videos=False for one MP4 per episode (also valid v3; fewer
files per camera, same path template and metadata).

v2.1 vs v3 (no mixture): We always emit v3 layout (videos/<camera>/chunk-000/file-XXX.mp4,
meta/episodes/ parquet, data/ parquet). v2.1 uses videos/chunk-000/<camera>/episode_*.mp4
and meta/episodes.jsonl. If something failed before, it was not due to mixing v2 and v3
paths; both one-file-per-episode and concatenated output are valid v3.

Video encoding: Exported videos are re-encoded to H.264 (avc1, yuv420p) so the LeRobot
dataset viewer and torchvision decode them correctly (only avc1 is reliably visible).
Metadata uses "video.codec": "avc1". Requires ffmpeg.

Usage:
    from robolab.core.export.lerobot_exporter import LeRobotExporter

    exporter = LeRobotExporter(
        robolab_output_dir="output/my_experiment",
        lerobot_output_dir="output/my_experiment_lerobot",
        robot_type="franka"
    )
    exporter.export()
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class LeRobotExporter:
    """Converts RoboLab HDF5 output to LeRobot v3.0 format."""

    def __init__(
        self,
        robolab_output_dir: str,
        lerobot_output_dir: str | None = None,
        robot_type: str = "franka",
        fps: float = 15.0,
        repo_id: str | None = None,
        concatenate_videos: bool = True,
        success_only: bool = False,
        require_complete_videos: bool = False,
        required_cameras: list[str] | tuple[str, ...] | None = None,
    ):
        """Initialize the LeRobot exporter.

        Args:
            robolab_output_dir: Path to the RoboLab output directory containing
                task folders with data.hdf5 files and videos.
            lerobot_output_dir: Path for LeRobot output. If None, creates a
                'lerobot' subdirectory in robolab_output_dir.
            robot_type: Robot type string for metadata (e.g., "franka", "droid").
            fps: Frames per second for the dataset.
            repo_id: Optional Hugging Face repo ID for the dataset.
            concatenate_videos: If True (default), merge all episode videos per camera
                into one MP4 (v3 convention). If False, write one MP4 per episode
                (file-000.mp4, file-001.mp4, ...); still v3, no ffmpeg needed.
            success_only: If True, export only demos whose HDF5 success attr is True.
            require_complete_videos: If True, skip demos without all required camera MP4s.
            required_cameras: Camera names required when require_complete_videos is True.
        """
        if not HAS_PYARROW:
            raise ImportError(
                "pyarrow is required for LeRobot export. "
                "Install with: pip install pyarrow"
            )

        self.robolab_dir = Path(robolab_output_dir)
        self.lerobot_dir = Path(lerobot_output_dir) if lerobot_output_dir else self.robolab_dir / "lerobot"
        # LeRobot does not support "franka"; save as "unknown" for compatibility
        self.robot_type = "unknown" if robot_type == "franka" else robot_type
        self.fps = fps
        self.repo_id = repo_id or f"robolab/{self.robolab_dir.name}"
        self.concatenate_videos = concatenate_videos
        self.success_only = success_only
        self.require_complete_videos = require_complete_videos
        self.required_cameras = tuple(required_cameras or ())

        # Statistics accumulators
        self._stats: dict[str, dict[str, list]] = {}
        self._tasks: list[dict] = []
        self._episodes_metadata: list[dict] = []

        # Data accumulators for parquet
        self._all_data_rows: list[dict] = []
        self._current_index = 0

        # Video info
        self._video_features: dict[str, dict] = {}
        self._video_files: list[tuple[str, str, int]] = []  # (camera, src_path, episode_idx)
        # After building concatenated videos: (camera_name, episode_idx) -> (file_index, from_ts, to_ts)
        self._video_timestamps: dict[tuple[str, int], tuple[int, float, float]] = {}

    def export(self) -> Path:
        """Export RoboLab data to LeRobot v3.0 format.

        Returns:
            Path to the created LeRobot dataset directory.
        """
        print(f"[LeRobotExporter] Exporting from: {self.robolab_dir}")
        print(f"[LeRobotExporter] Exporting to: {self.lerobot_dir}")

        # Create output directory structure
        self._create_directory_structure()

        # Find and process all HDF5 files
        hdf5_files = self._find_hdf5_files()
        if not hdf5_files:
            raise ValueError(f"No HDF5 files (data.hdf5 or run_*.hdf5) found in {self.robolab_dir}")

        print(f"[LeRobotExporter] Found {len(hdf5_files)} HDF5 files")

        # Process each HDF5 file
        for hdf5_path in hdf5_files:
            self._process_hdf5_file(hdf5_path)

        # Write all accumulated data
        self._write_data_parquet()
        self._build_concatenated_videos()
        self._write_episodes_metadata()
        self._write_tasks()
        self._write_modality()
        self._write_stats()
        self._write_info()

        print(f"[LeRobotExporter] Export complete: {self.lerobot_dir}")
        return self.lerobot_dir

    def _create_directory_structure(self):
        """Create the LeRobot v3.0 directory structure."""
        dirs = [
            self.lerobot_dir / "meta" / "episodes" / "chunk-000",
            self.lerobot_dir / "data" / "chunk-000",
            self.lerobot_dir / "videos",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def _find_hdf5_files(self) -> list[Path]:
        """Find all HDF5 files in the RoboLab output directory.

        Supports both legacy single-file format (data.hdf5) and multi-env
        per-run format (run_0.hdf5, run_1.hdf5, ...).
        """
        hdf5_files = []

        def _collect_from_dir(d: Path):
            # Legacy single file
            data_path = d / "data.hdf5"
            if data_path.exists():
                hdf5_files.append(data_path)
            # Multi-env per-run files, sorted by run index
            run_files = sorted(d.glob("run_*.hdf5"),
                               key=lambda p: int(p.stem.split("_")[1]))
            hdf5_files.extend(run_files)

        # Task subdirectories
        for task_dir in self.robolab_dir.iterdir():
            if task_dir.is_dir():
                _collect_from_dir(task_dir)

        # Root directory
        _collect_from_dir(self.robolab_dir)

        return hdf5_files

    def _process_hdf5_file(self, hdf5_path: Path):
        """Process a single HDF5 file and extract episode data."""
        task_dir = hdf5_path.parent
        task_name = task_dir.name if task_dir != self.robolab_dir else "default_task"

        # Detect multi-env per-run files (run_0.hdf5 → run_idx=0)
        stem = hdf5_path.stem
        if stem.startswith("run_"):
            run_idx = int(stem.split("_")[1])
        else:
            run_idx = None

        print(f"[LeRobotExporter] Processing: {hdf5_path}")

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"  Warning: No 'data' group in {hdf5_path}")
                return

            data_group = f["data"]

            # Find all demo groups
            demo_names = [k for k in data_group.keys() if k.startswith("demo_")]
            demo_names.sort(key=lambda x: int(x.split("_")[1]))

            print(f"  Found {len(demo_names)} episodes")

            for demo_name in demo_names:
                demo_group = data_group[demo_name]
                if self.success_only and not bool(demo_group.attrs.get("success", False)):
                    continue
                if self.require_complete_videos and not self._has_required_episode_videos(
                    task_dir, demo_name, run_idx=run_idx
                ):
                    continue
                episode_idx = len(self._episodes_metadata)

                # Extract episode data
                episode_data = self._extract_episode_data(demo_group, task_name)

                if episode_data:
                    # Find corresponding video files
                    self._find_episode_videos(task_dir, episode_idx, demo_name, run_idx=run_idx)

                    # Get task info
                    task_idx = self._get_or_create_task(task_name, task_dir=task_dir)

                    # Add episode metadata (tasks = instruction list, for compatibility with HF/LeRobot visualizer)
                    num_frames = len(episode_data)
                    task_entry = self._tasks[task_idx]
                    instruction = task_entry["task"]
                    instruction_2 = task_entry.get("instruction_2", instruction)
                    instruction_3 = task_entry.get("instruction_3", instruction)
                    self._episodes_metadata.append({
                        "episode_index": episode_idx,
                        "data/chunk_index": 0,
                        "data/file_index": 0,
                        "dataset_from_index": self._current_index,
                        "dataset_to_index": self._current_index + num_frames,
                        "length": num_frames,
                        "task_index": task_idx,
                        "success": bool(demo_group.attrs.get("success", False)),
                        "tasks": [instruction],
                    })

                    # Add data rows with episode context
                    num_frames_ep = len(episode_data)
                    for frame_idx, row in enumerate(episode_data):
                        row["episode_index"] = episode_idx
                        row["frame_index"] = frame_idx
                        row["index"] = self._current_index
                        row["task_index"] = task_idx
                        row["language_instruction"] = instruction
                        row["language_instruction_2"] = instruction_2
                        row["language_instruction_3"] = instruction_3
                        row["timestamp"] = frame_idx / self.fps
                        # LeRobot v3: next.done = True on last frame of episode (for visualizer)
                        row["next.done"] = frame_idx == num_frames_ep - 1
                        self._all_data_rows.append(row)
                        self._current_index += 1

                    # Update statistics
                    self._update_stats(episode_data)

    def _extract_episode_data(self, demo_group: h5py.Group, task_name: str) -> list[dict]:
        """Extract frame data from a demo group."""
        rows = []

        # Get number of samples
        num_samples = demo_group.attrs.get("num_samples", 0)
        if num_samples == 0:
            # Try to infer from actions shape
            if "actions" in demo_group:
                num_samples = demo_group["actions"].shape[0]

        if num_samples == 0:
            return rows

        # Extract actions
        actions = None
        if "actions" in demo_group:
            actions = np.array(demo_group["actions"])

        # Extract robot joint states
        joint_positions = None
        joint_velocities = None
        if "states" in demo_group and "articulation" in demo_group["states"]:
            robot_states = demo_group["states"]["articulation"].get("robot", {})
            if "joint_position" in robot_states:
                joint_positions = np.array(robot_states["joint_position"])
            if "joint_velocity" in robot_states:
                joint_velocities = np.array(robot_states["joint_velocity"])

        # Extract end-effector pose if available
        ee_position = None
        ee_orientation = None
        if "ee_pose" in demo_group:
            ee_group = demo_group["ee_pose"]
            if "position" in ee_group:
                ee_position = np.array(ee_group["position"])
            if "orientation" in ee_group:
                ee_orientation = np.array(ee_group["orientation"])

        # Build rows
        for i in range(num_samples):
            row = {}

            # Actions (required for LeRobot)
            if actions is not None and i < len(actions):
                row["action"] = actions[i].tolist()

            # Observation state (joint positions are commonly used)
            if joint_positions is not None and i < len(joint_positions):
                row["observation.state"] = joint_positions[i].tolist()

            # Optional: joint velocities
            if joint_velocities is not None and i < len(joint_velocities):
                row["observation.velocity"] = joint_velocities[i].tolist()

            # Optional: end-effector pose
            if ee_position is not None and i < len(ee_position):
                row["observation.ee_position"] = ee_position[i].tolist()
            if ee_orientation is not None and i < len(ee_orientation):
                row["observation.ee_orientation"] = ee_orientation[i].tolist()

            rows.append(row)

        return rows

    @staticmethod
    def _is_placeholder_instruction(instruction: str | None) -> bool:
        if instruction is None:
            return True
        value = str(instruction).strip()
        return value == "" or value.lower() in {
            "default task",
            "default instruction",
            "task",
            "instruction",
        }

    def _instruction_from_task_name(self, task_name: str) -> str:
        import re

        words = re.findall(r'[A-Z][a-z]*|[a-z]+', task_name.replace("Task", ""))
        return " ".join(words).lower().capitalize()

    def _task_metadata_by_name(self) -> dict[str, dict]:
        if hasattr(self, "_task_metadata_cache"):
            return self._task_metadata_cache
        metadata_path = Path(__file__).resolve().parents[2] / "tasks" / "_metadata" / "task_metadata.json"
        metadata: dict[str, dict] = {}
        if metadata_path.exists():
            try:
                entries = json.loads(metadata_path.read_text())
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("task_name"):
                            metadata[str(entry["task_name"])] = entry
            except (OSError, json.JSONDecodeError):
                metadata = {}
        self._task_metadata_cache = metadata
        return metadata

    def _get_instruction_variants_for_task(self, task_name: str, task_dir: Path | None = None) -> dict[str, str]:
        """Resolve RoboLab task instructions from task metadata.

        Training exports should use the task's canonical instruction variants,
        independent of which instruction type was used by a particular eval run.
        """
        metadata = self._task_metadata_by_name().get(task_name, {})
        variants: dict[str, str] = {}

        if isinstance(metadata.get("instruction_variants"), dict):
            variants.update({str(k): str(v) for k, v in metadata["instruction_variants"].items() if v})
        if metadata.get("instruction") and "default" not in variants:
            variants["default"] = str(metadata["instruction"])

        if not variants and task_dir is not None:
            env_cfg_path = task_dir / "env_cfg.json"
            if env_cfg_path.exists():
                try:
                    env_cfg = json.loads(env_cfg_path.read_text())
                    env_variants = env_cfg.get("_instruction_variants")
                    if isinstance(env_variants, dict):
                        variants.update({str(k): str(v) for k, v in env_variants.items() if v})
                    env_instruction = env_cfg.get("instruction")
                    if not self._is_placeholder_instruction(env_instruction):
                        variants.setdefault("default", str(env_instruction).strip())
                except (OSError, json.JSONDecodeError):
                    pass

        if not variants:
            from robolab.core.logging.results import load_episode_results

            try:
                results = load_episode_results(str(task_dir or self.robolab_dir))
            except Exception:
                results = []
            for result in results:
                if result.get("task") == task_name and "instruction" in result:
                    result_instruction = str(result["instruction"]).strip()
                    if not self._is_placeholder_instruction(result_instruction):
                        variants["default"] = result_instruction
                        break

        default_instruction = variants.get("default")
        if self._is_placeholder_instruction(default_instruction):
            default_instruction = self._instruction_from_task_name(task_name)
        vague_instruction = variants.get("vague") or default_instruction
        specific_instruction = variants.get("specific") or default_instruction

        return {
            "primary": str(default_instruction),
            "default": str(default_instruction),
            "vague": str(vague_instruction),
            "specific": str(specific_instruction),
        }

    def _load_camera_info_from_env_cfg(self, task_dir: Path) -> dict[str, dict]:
        """Load camera metadata from env_cfg.json if available.

        Returns:
            Dict mapping camera_name -> {"width": int, "height": int, "data_types": list}
        """
        env_cfg_path = task_dir / "env_cfg.json"
        if not env_cfg_path.exists():
            return {}
        try:
            with open(env_cfg_path) as f:
                cfg = json.load(f)
            cameras = {}
            scene = cfg.get("scene", {})
            for key, value in scene.items():
                if isinstance(value, dict):
                    class_type = value.get("class_type", "")
                    if class_type and "camera" in class_type.lower():
                        cameras[key] = {
                            "width": value.get("width", 640),
                            "height": value.get("height", 480),
                            "data_types": value.get("data_types", ["rgb"]),
                        }
            return cameras
        except (json.JSONDecodeError, KeyError):
            return {}

    def _episode_camera_video_paths(self, task_dir: Path, demo_name: str, run_idx: int | None = None) -> dict[str, Path]:
        demo_num = int(demo_name.split("_")[1])
        if run_idx is not None:
            pattern = f"*_{run_idx}_env{demo_num}__*.mp4"
        else:
            pattern = f"*_{demo_num}__*.mp4"
        paths = {}
        for video_file in task_dir.glob(pattern):
            stem = video_file.stem
            dunder_idx = stem.rfind("__")
            if dunder_idx == -1:
                continue
            cam_name = stem[dunder_idx + 2:]
            if video_file.exists() and video_file.stat().st_size > 0:
                paths[cam_name] = video_file
        return paths

    def _has_required_episode_videos(self, task_dir: Path, demo_name: str, run_idx: int | None = None) -> bool:
        paths = self._episode_camera_video_paths(task_dir, demo_name, run_idx=run_idx)
        if self.required_cameras:
            return all(camera in paths for camera in self.required_cameras)
        return bool(paths)

    def _find_episode_videos(self, task_dir: Path, episode_idx: int, demo_name: str, run_idx: int | None = None):
        """Find video files for an episode.

        For multi-env (run_idx is not None): demo_num is the env_id, videos are
        named {instruction}_{run_idx}_env{env_id}.mp4 or with __camera suffix.

        For legacy (run_idx is None): videos are named {instruction}_{demo_num}.mp4
        or with __camera suffix.
        """
        demo_num = int(demo_name.split("_")[1])

        # Load camera info from env_cfg.json for metadata
        if not hasattr(self, '_camera_info_cache'):
            self._camera_info_cache = {}
        if str(task_dir) not in self._camera_info_cache:
            self._camera_info_cache[str(task_dir)] = self._load_camera_info_from_env_cfg(task_dir)
        camera_info = self._camera_info_cache[str(task_dir)]

        # Build glob patterns based on whether this is a multi-env run
        if run_idx is not None:
            # Multi-env: demo_num is env_id, video pattern is *_{run_idx}_env{env_id}__*.mp4
            per_camera_pattern = f"*_{run_idx}_env{demo_num}__*.mp4"
        else:
            # Legacy: video pattern is *_{demo_num}__*.mp4
            per_camera_pattern = f"*_{demo_num}__*.mp4"

        # First, check for per-camera videos (double underscore pattern)
        per_camera_found = False
        for video_file in task_dir.glob(per_camera_pattern):
            # Extract camera name from {instruction}_{ep}__{camera_name}.mp4
            stem = video_file.stem
            dunder_idx = stem.rfind("__")
            if dunder_idx == -1:
                continue
            cam_name = stem[dunder_idx + 2:]
            feature_name = f"observation.images.{cam_name}"

            self._video_files.append((feature_name, str(video_file), episode_idx))
            per_camera_found = True

            if feature_name not in self._video_features:
                # Prefer env_cfg.json metadata, fall back to video probe
                cam_meta = camera_info.get(cam_name, {})
                if cam_meta:
                    width = cam_meta["width"]
                    height = cam_meta["height"]
                else:
                    width, height, _ = self._get_video_info(video_file)
                self._video_features[feature_name] = {
                    "dtype": "video",
                    "shape": [height, width, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": height,
                        "video.width": width,
                        "video.codec": "avc1",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": float(self.fps),
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }

        if per_camera_found:
            return

        # Fallback: combined viewport video
        if run_idx is not None:
            viewport_match = f"_{run_idx}_env{demo_num}_viewport"
        else:
            viewport_match = None
        for video_file in task_dir.glob("*.mp4"):
            name = video_file.stem
            if "_viewport" not in name:
                continue
            if run_idx is not None:
                matched = viewport_match in name
            else:
                matched = f"_{demo_num}" in name or name.endswith(f"_{demo_num}")
            if matched:
                camera_name = "observation.images.front"

                self._video_files.append((camera_name, str(video_file), episode_idx))

                if camera_name not in self._video_features:
                    width, height, fps = self._get_video_info(video_file)
                    if fps <= 0:
                        fps = self.fps
                    self._video_features[camera_name] = {
                        "dtype": "video",
                        "shape": [height, width, 3],
                        "names": ["height", "width", "channels"],
                        "info": {
                            "video.height": height,
                            "video.width": width,
                            "video.codec": "avc1",
                            "video.pix_fmt": "yuv420p",
                            "video.is_depth_map": False,
                            "video.fps": float(fps),
                            "video.channels": 3,
                            "has_audio": False,
                        },
                    }
                break

    def _get_video_info(self, video_path: Path) -> tuple[int, int, float]:
        """Get video dimensions and fps using OpenCV."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            return width, height, fps
        except Exception:
            return 640, 480, self.fps

    def _get_video_duration_seconds(self, video_path: Path) -> float:
        """Get video duration in seconds (for concatenation timestamp tracking)."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0 and n_frames >= 0:
                return n_frames / fps
            return 0.0
        except Exception:
            return 0.0

    def _reencode_to_avc1(self, src_path: Path, dst_path: Path, fps: float | None = None) -> bool:
        """Re-encode video to H.264 (avc1) yuv420p for LeRobot viewer / torchvision. Returns True on success."""
        fps = fps or self.fps
        cmd = [
            "ffmpeg", "-y",
            "-i", str(src_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-r", str(fps),
            "-an",
            str(dst_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _compute_observation_images_stats(self) -> dict[str, dict]:
        """Return lightweight placeholder image stats for each observation.images.* camera.

        Decoding all exported MP4s is expensive for RoboLab120-sized datasets and
        the StarVLA training loader computes its own numeric stats later. Keep
        LeRobot-compatible metadata here without scanning video pixels.
        """
        result = {}

        for camera_name in self._video_features:
            total_count = 0
            for camera, _, ep_idx in self._video_files:
                if camera != camera_name:
                    continue
                if 0 <= ep_idx < len(self._episodes_metadata):
                    total_count += self._episodes_metadata[ep_idx]["length"]

            result[camera_name] = {
                "min": [[[0.0]], [[0.0]], [[0.0]]],
                "max": [[[1.0]], [[1.0]], [[1.0]]],
                "mean": [[[0.5]], [[0.5]], [[0.5]]],
                "std": [[[0.5]], [[0.5]], [[0.5]]],
                "count": [total_count],
            }

        return result

    def _get_or_create_task(self, task_name: str, task_dir: Path | None = None) -> int:
        """Get task index, creating new task entry if needed."""
        instructions = self._get_instruction_variants_for_task(task_name, task_dir=task_dir)
        instruction = instructions["primary"]

        for i, task in enumerate(self._tasks):
            if task["task"] == instruction:
                return i

        task_idx = len(self._tasks)
        self._tasks.append({
            "task_index": task_idx,
            "task": instruction,
            "instruction_default": instructions["default"],
            "instruction_vague": instructions["vague"],
            "instruction_specific": instructions["specific"],
            "instruction_2": instructions["vague"],
            "instruction_3": instructions["specific"],
        })
        return task_idx

    def _update_stats(self, episode_data: list[dict]):
        """Update running statistics for all features (numeric and index-like)."""
        for row in episode_data:
            for key, value in row.items():
                if key == "next.done":
                    continue

                if isinstance(value, list):
                    arr = np.array(value)
                    if key not in self._stats:
                        self._stats[key] = {"values": []}
                    self._stats[key]["values"].append(arr)
                elif key in ("episode_index", "frame_index", "index", "task_index", "timestamp"):
                    # Index-like / scalar: store as 1-element array for stats
                    if key not in self._stats:
                        self._stats[key] = {"values": []}
                    self._stats[key]["values"].append(np.array([value]))

    def _compute_final_stats(self) -> dict:
        """Compute final statistics from accumulated values (with count)."""
        stats = {}

        for key, data in self._stats.items():
            if not data["values"]:
                continue

            all_values = np.stack(data["values"])
            n = len(data["values"])
            mean = all_values.mean(axis=0)
            std = all_values.std(axis=0)
            # Avoid NaN std when n<=1
            std = np.nan_to_num(std, nan=0.0, posinf=0.0, neginf=0.0)
            min_val = all_values.min(axis=0)
            max_val = all_values.max(axis=0)
            # Ensure list output (timestamp/index-like are 1d -> [x])
            def to_list(a):
                a = np.asarray(a)
                return a.tolist() if a.shape != () else [float(a)]

            stats[key] = {
                "mean": to_list(mean),
                "std": to_list(std),
                "min": to_list(min_val),
                "max": to_list(max_val),
                "count": [n],
            }

        return stats

    def _write_data_parquet(self):
        """Write all episode data to a single parquet file."""
        if not self._all_data_rows:
            print("  Warning: No data rows to write")
            return

        # Build schema based on first row and validate list lengths
        first_row = self._all_data_rows[0]
        schema_fields = []
        list_lengths = {}  # Track expected list lengths for validation

        for key, value in first_row.items():
            if key in ("episode_index", "frame_index", "index", "task_index"):
                schema_fields.append(pa.field(key, pa.int64()))
            elif key == "timestamp":
                schema_fields.append(pa.field(key, pa.float32()))
            elif key == "next.done":
                schema_fields.append(pa.field(key, pa.bool_()))
            elif isinstance(value, str):
                schema_fields.append(pa.field(key, pa.string()))
            elif isinstance(value, list):
                # Use variable-length list to match reference datasets (e.g. thanos, aloha)
                list_length = len(value)
                list_lengths[key] = list_length
                schema_fields.append(pa.field(key, pa.list_(pa.float32())))
            else:
                schema_fields.append(pa.field(key, pa.float32()))

        schema = pa.schema(schema_fields)

        # Validate all rows have consistent list lengths
        for row in self._all_data_rows:
            for key, expected_length in list_lengths.items():
                value = row.get(key)
                if isinstance(value, list) and len(value) != expected_length:
                    raise ValueError(
                        f"Inconsistent list length for '{key}': expected {expected_length}, "
                        f"got {len(value)}"
                    )

        # Convert rows to columnar format
        columns = {field.name: [] for field in schema}

        for row in self._all_data_rows:
            for key in columns:
                value = row.get(key)
                if value is None:
                    if key in ("episode_index", "frame_index", "index", "task_index"):
                        value = 0
                    elif key == "timestamp":
                        value = 0.0
                    elif key == "next.done":
                        value = False
                    elif key in ("language_instruction", "language_instruction_2", "language_instruction_3"):
                        value = ""
                    else:
                        # For list fields, create empty list of correct length
                        if key in list_lengths:
                            value = [0.0] * list_lengths[key]
                        else:
                            value = []
                columns[key].append(value)

        # Create table and write
        table = pa.table(columns, schema=schema)
        output_path = self.lerobot_dir / "data" / "chunk-000" / "file-000.parquet"
        pq.write_table(table, output_path)
        print(f"  Wrote {len(self._all_data_rows)} frames to {output_path}")

    def _write_episodes_metadata(self):
        """Write episode metadata to parquet."""
        if not self._episodes_metadata:
            return

        # Add video metadata columns (only point to existing files to avoid visualizer loading 404s).
        # v3: one MP4 per camera with concatenated episodes; from_timestamp/to_timestamp seek to segment.
        for ep in self._episodes_metadata:
            ep_idx = ep["episode_index"]
            for camera_name in self._video_features:
                ep[f"videos/{camera_name}/chunk_index"] = 0
                key = (camera_name, ep_idx)
                if key in self._video_timestamps:
                    file_index, from_ts, to_ts = self._video_timestamps[key]
                    ep[f"videos/{camera_name}/file_index"] = file_index
                    ep[f"videos/{camera_name}/from_timestamp"] = from_ts
                    ep[f"videos/{camera_name}/to_timestamp"] = to_ts
                else:
                    # No video for this episode/camera: use -1 so visualizer doesn't request missing file
                    ep[f"videos/{camera_name}/file_index"] = -1
                    ep[f"videos/{camera_name}/from_timestamp"] = 0.0
                    ep[f"videos/{camera_name}/to_timestamp"] = 0.0

        # Build schema (include success and tasks for HF/LeRobot visualizer compatibility)
        schema_fields = [
            pa.field("episode_index", pa.int64()),
            pa.field("data/chunk_index", pa.int64()),
            pa.field("data/file_index", pa.int64()),
            pa.field("dataset_from_index", pa.int64()),
            pa.field("dataset_to_index", pa.int64()),
            pa.field("length", pa.int64()),
            pa.field("task_index", pa.int64()),
            pa.field("success", pa.bool_()),
            pa.field("tasks", pa.list_(pa.string())),
        ]

        # Add video metadata fields
        for camera_name in self._video_features:
            # thanos uses double for video timestamps; use float64 for compatibility
            schema_fields.extend([
                pa.field(f"videos/{camera_name}/chunk_index", pa.int64()),
                pa.field(f"videos/{camera_name}/file_index", pa.int64()),
                pa.field(f"videos/{camera_name}/from_timestamp", pa.float64()),
                pa.field(f"videos/{camera_name}/to_timestamp", pa.float64()),
            ])

        schema = pa.schema(schema_fields)

        # Build columns with explicit type conversion
        columns = {field.name: [] for field in schema}
        for ep in self._episodes_metadata:
            for key in columns:
                default = False if key == "success" else ([] if key == "tasks" else (0.0 if "timestamp" in key else 0))
                value = ep.get(key, default)
                if key == "success":
                    value = bool(value)
                elif key == "tasks":
                    value = value if isinstance(value, list) else []
                elif "timestamp" in key:
                    value = float(value)
                else:
                    value = int(value)
                columns[key].append(value)

        table = pa.table(columns, schema=schema)
        output_path = self.lerobot_dir / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        pq.write_table(table, output_path)
        print(f"  Wrote {len(self._episodes_metadata)} episode metadata entries")

    def _write_tasks(self):
        """Write tasks to JSONL and parquet files."""
        # Write JSONL format
        jsonl_path = self.lerobot_dir / "meta" / "tasks.jsonl"
        with open(jsonl_path, "w") as f:
            for task in self._tasks:
                f.write(json.dumps(task) + "\n")

        # Also write parquet format (some visualizers prefer this)
        if self._tasks:
            schema = pa.schema([
                pa.field("task_index", pa.int64()),
                pa.field("task", pa.string()),
            ])
            columns = {
                "task_index": [t["task_index"] for t in self._tasks],
                "task": [t["task"] for t in self._tasks],
            }
            table = pa.table(columns, schema=schema)
            parquet_path = self.lerobot_dir / "meta" / "tasks.parquet"
            pq.write_table(table, parquet_path)

        print(f"  Wrote {len(self._tasks)} tasks")


    def _write_modality(self):
        """Write StarVLA/world_lerobot-compatible modality metadata."""
        video = {
            "exterior_image_1_left": {"original_key": "observation.images.over_shoulder_left_camera"},
            "exterior_image_2_left": {"original_key": "observation.images.over_shoulder_right_camera"},
            "wrist_image_left": {"original_key": "observation.images.wrist_cam"},
        }
        available_video_keys = set(self._video_features.keys())
        video = {
            key: value
            for key, value in video.items()
            if value["original_key"] in available_video_keys
        }
        modality = {
            "state": {
                "joint_position": {
                    "start": 0,
                    "end": 7,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": "observation.state",
                },
                "gripper_position": {
                    "start": 12,
                    "end": 13,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": "observation.state",
                },
            },
            "action": {
                "joint_position": {
                    "start": 0,
                    "end": 7,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": "action",
                },
                "gripper_position": {
                    "start": 7,
                    "end": 8,
                    "absolute": True,
                    "dtype": "float32",
                    "original_key": "action",
                },
            },
            "video": video,
            "annotation": {
                "language.language_instruction": {"original_key": "language_instruction"},
                "language.language_instruction_2": {"original_key": "language_instruction_2"},
                "language.language_instruction_3": {"original_key": "language_instruction_3"},
            },
        }
        output_path = self.lerobot_dir / "meta" / "modality.json"
        with open(output_path, "w") as f:
            json.dump(modality, f, indent=2)
        print("  Wrote modality.json")

    def _write_stats(self):
        """Write statistics to JSON file."""
        stats = self._compute_final_stats()
        # Add observation.images.* stats (min/max/mean/std/count in nested format)
        for camera_name, img_stats in self._compute_observation_images_stats().items():
            stats[camera_name] = img_stats
        output_path = self.lerobot_dir / "meta" / "stats.json"
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Wrote statistics for {len(stats)} features")

    def _write_info(self):
        """Write dataset info.json file."""
        # Determine feature shapes from data
        features = {}
        fps_int = int(round(self.fps))

        if self._all_data_rows:
            first_row = self._all_data_rows[0]

            for key, value in first_row.items():
                if key in ("episode_index", "frame_index", "index", "task_index"):
                    features[key] = {
                        "dtype": "int64",
                        "shape": [1],
                        "names": None,
                        "fps": fps_int,
                    }
                elif key == "timestamp":
                    features[key] = {
                        "dtype": "float32",
                        "shape": [1],
                        "names": None,
                        "fps": fps_int,
                    }
                elif key == "next.done":
                    features[key] = {
                        "dtype": "bool",
                        "shape": [1],
                        "names": None,
                        "fps": fps_int,
                    }
                elif isinstance(value, str):
                    features[key] = {
                        "dtype": "string",
                        "shape": [1],
                        "names": None,
                        "fps": fps_int,
                    }
                elif isinstance(value, list):
                    features[key] = {
                        "dtype": "float32",
                        "shape": [len(value)],
                        "names": self._get_feature_names(key, len(value)),
                        "fps": fps_int,
                    }

        # Add video features
        features.update(self._video_features)

        info = {
            "codebase_version": "v3.0",
            "robot_type": self.robot_type,
            "total_episodes": len(self._episodes_metadata),
            "total_frames": len(self._all_data_rows),
            "total_tasks": len(self._tasks),
            "total_videos": len({c for c, _, _ in self._video_files}) if self.concatenate_videos else len(self._video_files),
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": fps_int,
            "splits": {
                "train": f"0:{len(self._episodes_metadata)}"
            },
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": features,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 500,
        }

        output_path = self.lerobot_dir / "meta" / "info.json"
        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)
        print(f"  Wrote info.json")

    def _get_feature_names(self, key: str, length: int) -> dict | list | None:
        """Get human-readable names for feature dimensions."""
        if "joint" in key.lower() or key == "observation.state":
            return {"motors": [f"joint_{i}" for i in range(length)]}
        elif key == "action":
            # Typical robot action: 6 DOF arm + gripper
            if length == 7:
                return {"motors": ["x", "y", "z", "rx", "ry", "rz", "gripper"]}
            elif length == 8:
                return {"motors": ["j0", "j1", "j2", "j3", "j4", "j5", "j6", "gripper"]}
            else:
                return {"motors": [f"action_{i}" for i in range(length)]}
        elif "position" in key.lower():
            if length == 3:
                return ["x", "y", "z"]
            elif length == 7:  # pose
                return ["x", "y", "z", "qx", "qy", "qz", "qw"]
        elif "orientation" in key.lower():
            if length == 4:
                return ["qx", "qy", "qz", "qw"]

        return None

    def _build_concatenated_videos(self):
        """Build video files: either one concatenated MP4 per camera (default) or one MP4 per episode (v3 both)."""
        if self.concatenate_videos:
            self._build_concatenated_videos_merged()
        else:
            self._build_videos_one_per_episode()

    def _build_videos_one_per_episode(self):
        """Write one MP4 per episode (file-000.mp4, file-001.mp4, ...), re-encoded to avc1."""
        for camera_name, src_path, episode_idx in self._video_files:
            path = Path(src_path)
            if not path.exists():
                continue
            camera_dir = self.lerobot_dir / "videos" / camera_name / "chunk-000"
            camera_dir.mkdir(parents=True, exist_ok=True)
            dst_path = camera_dir / f"file-{episode_idx:03d}.mp4"
            if self._reencode_to_avc1(path, dst_path):
                duration = self._get_video_duration_seconds(dst_path)
            else:
                shutil.copy2(path, dst_path)
                duration = self._get_video_duration_seconds(path)
            self._video_timestamps[(camera_name, episode_idx)] = (
                episode_idx,
                0.0,
                duration,
            )
        if self._video_files:
            print(f"  Wrote {len(self._video_files)} episode videos (one file per episode, avc1)")

    def _build_concatenated_videos_merged(self):
        """One concatenated MP4 per camera (avc1); episode segment via from_timestamp/to_timestamp. Requires ffmpeg."""
        import shlex

        for camera_name in self._video_features:
            # All (src_path, episode_idx) for this camera, sorted by episode index
            entries = sorted(
                [(p, e) for c, p, e in self._video_files if c == camera_name],
                key=lambda x: x[1],
            )
            if not entries:
                continue

            camera_dir = self.lerobot_dir / "videos" / camera_name / "chunk-000"
            camera_dir.mkdir(parents=True, exist_ok=True)
            out_path = camera_dir / "file-000.mp4"

            valid = [(Path(p), ep_idx) for p, ep_idx in entries if Path(p).exists()]
            if not valid:
                continue

            # Re-encode each source to avc1 (same codec) so concat -c copy works and output is LeRobot-compatible
            temp_avc1_dir = tempfile.mkdtemp()
            try:
                temp_paths = []
                cumulative_ts = 0.0
                for path, ep_idx in valid:
                    temp_f = Path(temp_avc1_dir) / f"ep_{ep_idx:04d}.mp4"
                    if self._reencode_to_avc1(path, temp_f):
                        temp_paths.append((temp_f, ep_idx))
                        duration = self._get_video_duration_seconds(temp_f)
                    else:
                        temp_paths.append((path, ep_idx))
                        duration = self._get_video_duration_seconds(path)
                    self._video_timestamps[(camera_name, ep_idx)] = (
                        0,
                        cumulative_ts,
                        cumulative_ts + duration,
                    )
                    cumulative_ts += duration

                # Concat avc1 segments with -c copy
                concat_path = Path(temp_avc1_dir) / "concat_list.txt"
                with open(concat_path, "w") as f:
                    for t, _ in temp_paths:
                        f.write(f"file {shlex.quote(str(t.absolute()))}\n")
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat", "-safe", "0",
                    "-i", str(concat_path),
                    "-c", "copy",
                    str(out_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                if result.returncode != 0:
                    print(
                        f"  Warning: ffmpeg concat failed for {camera_name}: "
                        f"{result.stderr[:500] if result.stderr else result.stdout}"
                    )
                    first_path, first_ep = valid[0]
                    if temp_paths and temp_paths[0][0].exists():
                        shutil.copy2(temp_paths[0][0], out_path)
                    else:
                        shutil.copy2(first_path, out_path)
                    for (cam, ep_idx) in list(self._video_timestamps.keys()):
                        if cam == camera_name and ep_idx != first_ep:
                            del self._video_timestamps[(cam, ep_idx)]
                    self._video_timestamps[(camera_name, first_ep)] = (
                        0, 0.0, self._get_video_duration_seconds(out_path)
                    )
                else:
                    print(f"  Concatenated {len(valid)} videos (avc1) -> {out_path.relative_to(self.lerobot_dir)}")
            finally:
                try:
                    shutil.rmtree(temp_avc1_dir, ignore_errors=True)
                except OSError:
                    pass


def export_to_lerobot(
    robolab_output_dir: str,
    lerobot_output_dir: str | None = None,
    robot_type: str = "franka",
    fps: float = 15.0,
    concatenate_videos: bool = True,
    success_only: bool = False,
    require_complete_videos: bool = False,
    required_cameras: list[str] | tuple[str, ...] | None = None,
) -> Path:
    """Convenience function to export RoboLab output to LeRobot format.

    Args:
        robolab_output_dir: Path to RoboLab output directory.
        lerobot_output_dir: Path for LeRobot output (optional).
        robot_type: Robot type string for metadata.
        fps: Frames per second.
        concatenate_videos: If True, one MP4 per camera (concatenated); if False, one MP4 per episode.

    Returns:
        Path to the created LeRobot dataset.
    """
    exporter = LeRobotExporter(
        robolab_output_dir=robolab_output_dir,
        lerobot_output_dir=lerobot_output_dir,
        robot_type=robot_type,
        fps=fps,
        concatenate_videos=concatenate_videos,
        success_only=success_only,
        require_complete_videos=require_complete_videos,
        required_cameras=required_cameras,
    )
    return exporter.export()
