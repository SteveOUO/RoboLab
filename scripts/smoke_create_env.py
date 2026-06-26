# SPDX-License-Identifier: CC-BY-NC-4.0
# isort: skip_file

import argparse
import cv2  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="MarkerInMugTask")
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.save_videos = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.registrations.droid_jointpos.auto_env_registrations import auto_register_droid_envs  # noqa: E402

print(f"[smoke] registering task={args.task}", flush=True)
auto_register_droid_envs(task=args.task)

print(f"[smoke] creating env task={args.task} num_envs={args.num_envs} cameras={args.enable_cameras}", flush=True)
env, env_cfg = create_env(
    args.task,
    device=args.device,
    num_envs=args.num_envs,
    use_fabric=True,
)

print(f"[smoke] created env instruction={env_cfg.instruction!r}", flush=True)
end_episode(env)
simulation_app.close()
print("[smoke] ok", flush=True)
