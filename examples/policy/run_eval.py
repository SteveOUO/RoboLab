# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

"""
Run policy evaluation across multiple tasks.

This script runs policy evaluation on multiple registered tasks, supporting various
policy backends (pi0, etc.) with options for subtask tracking and result logging.

Supports multi-env: each "run" spawns num_envs parallel episodes.
Total episodes = num_runs * num_envs.

Usage:
    Run on all registered tasks:
    $ python run_eval.py

    Run on specific tasks:
    $ python run_eval.py --task BananaInBowlTask RubiksCubeTask

    Run on a tag:
    $ python run_eval.py --tag spatial

    Use specific policy:
    $ python run_eval.py --policy pi05

    Run multiple episodes with 2 parallel envs:
    $ python run_eval.py --num-runs 2 --num_envs 4

Output:
    Results are saved to: output/<output_folder_name>/
"""

import argparse
import cv2 # Must import this before isaaclab. Do not remove
import os
import traceback
import sys
from isaaclab.app import AppLauncher
from robolab.constants import DEFAULT_TASK_SUBFOLDERS # noqa
from robolab.core.logging.run_paths import default_run_name, install_client_log, resolve_output_dir, utc8_now # noqa

# add argparse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs='+', default=None,
                       help="List of tasks to evaluate on ")
parser.add_argument("--tag", nargs='+', default=None,
                       help="List of tags of tasks to evaluate on ")
parser.add_argument("--task-dirs", nargs='+', default=DEFAULT_TASK_SUBFOLDERS,
                       help="List of task directories to evaluate on")
parser.add_argument("--policy",
                    choices=["pi0", "pi0_fast", "paligemma", "paligemma_fast", "pi05", "gr00t", "dreamzero", "smartworld", "cosmos3", "molmo", "openvla", "openvla_oft"], default="pi05",
                       help="Action-prediction backend to use (default: pi05)")
parser.add_argument("--num-runs", "--num_runs", type=int, default=1,
                       help="Number of sequential runs per task (default: 1). Total episodes = num_runs * num_envs. Prefer increasing --num_envs for more episodes. Only increase --num-runs if you run out of GPU memory with the desired num_envs.")
parser.add_argument("--enable-subtask", "--enable_subtask", action="store_true",
                       help="Enable subtask progress checking (default: False)")
parser.add_argument("--record-image-data", "--record_image_data", action="store_true",
                       help="Enable proprio image data recording (default: False)")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                       help="Output folder name under /robolab/output. Default is <timestamp>_<policy>. If you provide the output folder name for a previous run, the script will skip the tasks and episodes that have already been run.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                       help="Verbose output (default: False)")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true",
                       help="Debug output (default: False)")
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                       help="Remote host for policy server (default: localhost)")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                       help="Remote port for policy server (default: 8000)")
parser.add_argument("--openpi-action-mode", "--openpi_action_mode", type=str, default="joint_velocity", choices=["joint_velocity", "joint_position"],
                       help="OpenPI DROID action semantics. Use joint_velocity for pi05_droid, joint_position for jointpos checkpoints.")
parser.add_argument("--openpi-control-dt", "--openpi_control_dt", type=float, default=1.0 / 15.0,
                       help="Control dt used to integrate OpenPI joint_velocity actions into joint-position targets.")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                       help="Full WebSocket URI for policy server, e.g. wss://host.lepton.run. "
                            "Overrides --remote-host and --remote-port when set.")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                       help="Number of actions to execute from each predicted chunk before requesting a new one. "
                            "If omitted, each inference client uses its own default. "
                            "Must match the model's action_horizon for best performance.")
# DreamZero-specific flags (silently ignored by other backends via create_client kwarg filtering)
parser.add_argument("--dz-binarize-gripper", "--dz_binarize_gripper", action="store_true",
                    help="[DreamZero] Re-enable gripper binarization at 0.5 threshold (ablation; default: off)")
parser.add_argument("--dz-resize", "--dz_resize", type=str, default="area", choices=["area", "linear", "pad"],
                    help="[DreamZero] Image resize method: 'area' (default, INTER_AREA), 'linear' (INTER_LINEAR), "
                         "or 'pad' (aspect-preserving letterbox). Note: 'area'/'linear' change aspect ratio if "
                         "source differs from 180x320 target.")
parser.add_argument("--remote-token", "--remote_token", type=str, default=None,
                    help="Bearer token for authenticated endpoints (e.g. Lepton). "
                         "Falls back to DREAMZERO_API_TOKEN env var.")
parser.add_argument("--dz-cam2", "--dz_cam2", type=str, default="black",
                    choices=["black", "right", "head", "duplicate"],
                    help="[DreamZero] Second exterior camera: 'black' (default, matches training dropout), "
                         "'right' (over-shoulder), 'head' (front overhead), 'duplicate' (copy of left)")
parser.add_argument("--instruction-type", "--instruction_type", type=str, default="default",
                       help="Which instruction variant to use when a task defines multiple (default, vague, specific, etc.)")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save: 'all' (sensor + viewport), 'viewport' only, 'sensor' only, or 'none' (default: all)")
parser.add_argument("--max-episode-steps", "--max_episode_steps", type=int, default=None,
                    help="Optional cap on environment steps per episode for short smoke/debug runs.")
parser.add_argument("--randomize-background", "--randomize_background", action="store_true",
                    help="Sample a random non-default background per task at registration time. "
                         "Each registered env gets one fixed background; the chosen texture is "
                         "recorded in the per-task env_cfg.json.")
parser.add_argument("--background-seed", "--background_seed", type=int, default=None,
                    help="Seed for reproducible per-task background sampling. Used with --randomize-background.")
# parse the arguments
args_cli, _= parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
if args_cli.output_folder_name is None:
    args_cli.output_folder_name = default_run_name(
        args_cli.policy,
        instruction_type=args_cli.instruction_type,
    )
eval_started_at = utc8_now()
install_client_log(args_cli.output_folder_name)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from robolab.constants import set_output_dir # noqa
from robolab.core.environments.runtime import create_env # noqa
from robolab.eval import create_client, run_episode, summarize_run # noqa
from robolab.core.environments.factory import get_envs # noqa
from robolab.core.utils.print_utils import print_experiment_summary # noqa
from robolab.core.logging.results import check_all_episodes_complete, check_run_complete # noqa
from robolab.core.logging.results import init_experiment, summarize_experiment_results # noqa
from robolab.core.logging.result_summary import build_result_summary, write_result_yaml # noqa
import robolab.constants # noqa

# Update robolab.constants module settings from command line arguments
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug


def _read_smartworld_server_metadata() -> dict:
    from robolab_policy_client.smartworld import SmartWorldWebsocketClient

    client = SmartWorldWebsocketClient(args_cli.remote_host, args_cli.remote_port)
    try:
        metadata = client.get_server_metadata()
    finally:
        client.close()
    return metadata


# Run automatic factory generation before main
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs # noqa
registration_kwargs = {
    "task_dirs": args_cli.task_dirs,
    "task": args_cli.task,
    "randomize_background": args_cli.randomize_background,
    "background_seed": args_cli.background_seed,
}

if args_cli.policy == "smartworld":
    from robolab.registrations.droid_jointpos.camera_presets import WRIST_LEFT_RIGHT, with_camera_resolution # noqa
    smartworld_metadata = _read_smartworld_server_metadata()
    if "image_height" not in smartworld_metadata or "image_width" not in smartworld_metadata:
        raise ValueError(
            "SmartWorld server metadata must provide image_height and image_width before RoboLab env registration."
        )
    image_height = int(smartworld_metadata["image_height"])
    image_width = int(smartworld_metadata["image_width"])
    cameras, robot_cfg = with_camera_resolution(WRIST_LEFT_RIGHT, height=image_height, width=image_width)
    registration_kwargs["cameras"] = cameras
    registration_kwargs["robot_cfg"] = robot_cfg
    print(f"[RoboLab] SmartWorld camera render size: {image_height}x{image_width}")
elif args_cli.policy == "cosmos3":
    from robolab.registrations.droid_jointpos.camera_presets import WRIST_LEFT_RIGHT # noqa
    registration_kwargs["cameras"] = WRIST_LEFT_RIGHT
elif args_cli.policy == "dreamzero" and args_cli.dz_cam2 in ("right", "head"):
    from robolab.registrations.droid_jointpos.camera_presets import WRIST_LEFT_RIGHT_HEAD # noqa
    registration_kwargs["cameras"] = WRIST_LEFT_RIGHT_HEAD

auto_register_droid_envs(**registration_kwargs)

def main():
    """Main function."""
    output_dir = str(resolve_output_dir(args_cli.output_folder_name))
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
        filter_str = f"tasks: {', '.join(args_cli.task)}"
    elif args_cli.tag:
        task_envs = get_envs(tag=args_cli.tag)
        filter_str = f"tags: {', '.join(args_cli.tag)}"
    else:
        task_envs = get_envs()
        filter_str = "all"

    num_envs = args_cli.num_envs
    num_runs = args_cli.num_runs
    total_episodes = num_runs * num_envs

    print_experiment_summary(
        task_envs=task_envs,
        filter_str=filter_str,
        num_envs=num_envs,
        num_episodes=total_episodes,
        policy=args_cli.policy,
        instruction_type=args_cli.instruction_type,
        output_dir=output_dir,
    )

    episode_results_file, episode_results = init_experiment(output_dir)

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        if check_all_episodes_complete(episode_results=episode_results, env_name=task_env, num_episodes=total_episodes):
            print(f"\033[96m[RoboLab] Task `{task_env}` already done. Skipping.\033[0m")
            continue

        env, env_cfg = create_env(task_env,
            device=args_cli.device,
            num_envs=num_envs,
            use_fabric=True,
            instruction_type=args_cli.instruction_type,
            policy=args_cli.policy)

        # Construct the inference client once per task; reuse across runs.
        # CLI values of None are filtered out by create_client so the
        # client's own defaults apply.
        client = create_client(
            args_cli.policy,
            remote_host=args_cli.remote_host,
            remote_port=args_cli.remote_port,
            remote_uri=args_cli.remote_uri,
            open_loop_horizon=args_cli.open_loop_horizon,
            openpi_action_mode=args_cli.openpi_action_mode,
            openpi_control_dt=args_cli.openpi_control_dt,
            api_token=args_cli.remote_token,
            binarize_gripper=args_cli.dz_binarize_gripper,
            resize=args_cli.dz_resize,
            cam2_source=args_cli.dz_cam2,
            experiment_name=args_cli.output_folder_name,
            return_viz=not args_cli.headless,
        )

        for run_idx in range(num_runs):

            # Check if all episodes in this run are already complete
            run_episode_ids = [run_idx * num_envs + eid for eid in range(num_envs)]
            if all(check_run_complete(episode_results=episode_results, env_name=task_env, episode=ep_id) for ep_id in run_episode_ids):
                print(f"\033[96m[RoboLab] Task `{task_env}` run `{run_idx}` already done. Skipping.\033[0m")
                continue

            # Policy
            if args_cli.instruction_type != "default":
                run_name = task_env + f"_{args_cli.instruction_type}_{run_idx}"
            else:
                run_name = task_env + f"_{run_idx}"
            print(f"\033[96m[RoboLab] Running {run_name}: '{env_cfg.instruction}' (run {run_idx}, {num_envs} envs)\033[0m")

            env_results, msgs, timing = run_episode(env=env,
                        env_cfg=env_cfg,
                        episode=run_idx,
                        client=client,
                        save_videos=args_cli.save_videos,
                        video_mode=args_cli.video_mode,
                        max_episode_steps=args_cli.max_episode_steps,
                        headless=args_cli.headless)

            episode_results = summarize_run(
                env_results=env_results,
                msgs=msgs,
                timing=timing,
                env=env,
                env_cfg=env_cfg,
                num_envs=num_envs,
                run_idx=run_idx,
                run_name=run_name,
                task_env=task_env,
                scene_output_dir=scene_output_dir,
                policy=args_cli.policy,
                instruction_type=args_cli.instruction_type,
                episode_results=episode_results,
                episode_results_file=episode_results_file,
                enable_subtask_progress=robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING,
            )

            # Reset eval state for next run (unfreeze all envs)
            env.reset_eval_state()

        env.close()

    # This will print the results to the terminal, summarized.
    # Alternatively, you can run `python analysis/read_results.py <output_dir>` to read the results from the file.
    summarize_experiment_results(episode_results, show_timing=True)
    result_summary = build_result_summary(
        episode_results=episode_results,
        output_dir=output_dir,
        run_name=args_cli.output_folder_name,
        args=args_cli,
        started_at=eval_started_at,
        filter_str=filter_str,
    )
    write_result_yaml(result_summary)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
