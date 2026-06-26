# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Render RoboLab HDF5 recorder states into MP4 videos.

This is intended for post-hoc inspection of evaluation runs whose output folders
contain HDF5 recorder files but no RGB videos. It does not replay actions.
Instead, each recorded state is written back into IsaacSim and rendered through
the task cameras.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import cv2  # Must import this before isaaclab. Do not remove.
import h5py
import torch
from isaaclab.app import AppLauncher
from tqdm import tqdm

from robolab.constants import PACKAGE_DIR, set_output_dir  # noqa


parser = argparse.ArgumentParser(description="Render RoboLab HDF5 states to MP4.")
parser.add_argument("--manifest", type=Path, required=True, help="render_manifest.json from the failed-task report.")
parser.add_argument("--limit-tasks", type=int, default=None, help="Render only the first N tasks.")
parser.add_argument("--limit-videos", type=int, default=None, help="Render only the first N manifest entries.")
parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames per video.")
parser.add_argument("--frame-stride", type=int, default=1, help="Stride over recorded states.")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing MP4 files.")
parser.add_argument("--video-scale", type=float, default=0.5, help="Scale applied before writing combined sensor view.")
parser.add_argument("--instruction-type", type=str, default="specific")
parser.add_argument(
    "--camera-preset",
    type=str,
    default="WRIST_LEFT",
    help="Camera preset from robolab.registrations.droid_jointpos.camera_presets.",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.droid_jointpos import camera_presets  # noqa: E402
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs  # noqa: E402


def _camera_preset(name: str):
    try:
        preset = getattr(camera_presets, name)
    except AttributeError as exc:
        available = sorted(k for k in dir(camera_presets) if k.isupper())
        raise ValueError(f"Unknown camera preset {name!r}. Available presets: {available}") from exc
    if not isinstance(preset, list):
        raise ValueError(f"Camera preset {name!r} is not a list of camera configs.")
    return preset


auto_register_droid_envs(cameras=_camera_preset(args_cli.camera_preset))


def _load_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid manifest entries in {path}")
    return entries


def _state_tensor(dataset, step: int, device: str) -> torch.Tensor:
    return torch.as_tensor(dataset[step], device=device).unsqueeze(0)


def _state_from_hdf5(demo_group, step: int, device: str) -> dict:
    state = {"articulation": {}, "deformable_object": {}, "rigid_object": {}}

    states = demo_group["states"]
    if "articulation" in states:
        for asset_name, asset_group in states["articulation"].items():
            state["articulation"][asset_name] = {
                "root_pose": _state_tensor(asset_group["root_pose"], step, device),
                "root_velocity": _state_tensor(asset_group["root_velocity"], step, device),
                "joint_position": _state_tensor(asset_group["joint_position"], step, device),
                "joint_velocity": _state_tensor(asset_group["joint_velocity"], step, device),
            }
    if "deformable_object" in states:
        for asset_name, asset_group in states["deformable_object"].items():
            state["deformable_object"][asset_name] = {
                "nodal_position": _state_tensor(asset_group["nodal_position"], step, device),
                "nodal_velocity": _state_tensor(asset_group["nodal_velocity"], step, device),
            }
    if "rigid_object" in states:
        for asset_name, asset_group in states["rigid_object"].items():
            state["rigid_object"][asset_name] = {
                "root_pose": _state_tensor(asset_group["root_pose"], step, device),
                "root_velocity": _state_tensor(asset_group["root_velocity"], step, device),
            }
    return state


def _num_steps(demo_group) -> int:
    return int(demo_group["states"]["articulation"]["robot"]["joint_position"].shape[0])


def _entry_video_paths(entry: dict) -> dict[str, Path]:
    if "videos" in entry:
        return {str(camera): Path(path) for camera, path in entry["videos"].items()}
    if "video" in entry:
        return {"combined_image": Path(entry["video"])}
    raise ValueError(f"Manifest entry must contain 'video' or 'videos': {entry}")


def _render_entry(env, entry: dict, *, frame_stride: int, max_frames: int | None, overwrite: bool, video_scale: float) -> None:
    video_paths = _entry_video_paths(entry)
    if all(path.exists() for path in video_paths.values()) and not overwrite:
        print(f"[skip] {next(iter(video_paths.values())).parent} ({len(video_paths)} videos exist)")
        return

    for video_path in video_paths.values():
        video_path.parent.mkdir(parents=True, exist_ok=True)
    hdf5_path = Path(entry["hdf5"])
    demo_name = entry["demo"]

    env.reset()
    env_ids = torch.tensor([0], dtype=torch.long, device=env.device)
    fps = 1 / (env.cfg.sim.render_interval * env.cfg.sim.dt)
    writers = {
        camera: VideoWriter(str(video_path), fps=fps)
        for camera, video_path in video_paths.items()
        if overwrite or not video_path.exists()
    }

    try:
        with h5py.File(hdf5_path, "r") as f:
            demo_group = f["data"][demo_name]
            total_steps = _num_steps(demo_group)
            steps = range(0, total_steps, frame_stride)
            if max_frames is not None:
                steps = list(steps)[:max_frames]

            for step in tqdm(steps, desc=f"{entry['task']} {demo_name}", leave=False):
                state = _state_from_hdf5(demo_group, int(step), env.device)
                env.scene.reset_to(state, env_ids=env_ids, is_relative=True)
                env.sim.render()
                env.scene.update(env.cfg.sim.dt)
                obs = env.observation_manager.compute(update_history=False)
                frames = unpack_image_obs(obs, scale=video_scale, env_id=0)
                for camera, writer in writers.items():
                    if camera not in frames:
                        available = sorted(k for k in frames.keys() if k != "combined_image")
                        raise KeyError(f"Camera {camera!r} not found in observations. Available cameras: {available}")
                    writer.write(frames[camera])
    finally:
        for writer in writers.values():
            writer.release()


def main() -> None:
    entries = _load_manifest(args_cli.manifest)
    if args_cli.limit_videos is not None:
        entries = entries[: args_cli.limit_videos]
    if args_cli.limit_tasks is not None:
        task_order = []
        for entry in entries:
            if entry["task"] not in task_order:
                task_order.append(entry["task"])
        allowed = set(task_order[: args_cli.limit_tasks])
        entries = [entry for entry in entries if entry["task"] in allowed]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_task[entry["task"]].append(entry)

    output_root = args_cli.manifest.resolve().parent
    tmp_name = f"_isaacsim_render_tmp_{args_cli.manifest.stem}_{os.getpid()}"
    set_output_dir(str(output_root / tmp_name))

    for task, task_entries in by_task.items():
        task_envs = get_envs(task=task)
        if task not in task_envs:
            raise ValueError(f"Manifest task {task} is not a registered RoboLab env. Registered matches: {task_envs}")
        print(f"[task] {task}: {len(task_entries)} videos")
        env, _ = create_env(
            task,
            device=args_cli.device,
            num_envs=1,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy="cosmos3",
        )
        try:
            for entry in task_entries:
                _render_entry(
                    env,
                    entry,
                    frame_stride=args_cli.frame_stride,
                    max_frames=args_cli.max_frames,
                    overwrite=args_cli.overwrite,
                    video_scale=args_cli.video_scale,
                )
        finally:
            env.close()

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Terminated with error: {exc}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
