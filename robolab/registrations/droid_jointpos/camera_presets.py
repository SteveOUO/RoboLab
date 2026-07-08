# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""
Droid-specific camera preset bundles for the policy's image observations.

Each preset is a list of camera config classes that feed both the scene
(``camera_cfg=``) and the image observation group (via
``generate_image_obs_from_cameras``). Viewport-only cameras (e.g., the
third-person mirrored view used for video recording) are attached separately
inside the registration function and are not listed here.

Callers pass one of these lists directly to the registration function:

    from robolab.registrations.droid_jointpos.camera_presets import WRIST_LEFT_RIGHT_HEAD
    auto_register_droid_envs(cameras=WRIST_LEFT_RIGHT_HEAD)
"""

import copy
import dataclasses

from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from robolab.robots.droid import DroidCfg, WristCameraCfg
from robolab.variations.camera import (
    HeadCameraCfg,
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
)

WRIST = [WristCameraCfg]

WRIST_LEFT = [
    OverShoulderLeftCameraCfg,
    WristCameraCfg,
]

WRIST_RIGHT = [
    OverShoulderRightCameraCfg,
    WristCameraCfg,
]


WRIST_LEFT_RIGHT = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    WristCameraCfg,
]


WRIST_LEFT_RIGHT_HEAD = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
    HeadCameraCfg,
    WristCameraCfg,
]

LEFT_RIGHT = [
    OverShoulderLeftCameraCfg,
    OverShoulderRightCameraCfg,
]


def _camera_cfg_class_with_resolution(camera_cfg_cls: type, *, height: int, width: int) -> type:
    annotations = {}
    camera_attrs = {"__annotations__": annotations}
    found_camera = False
    for field in dataclasses.fields(camera_cfg_cls):
        value = field.default
        if value is dataclasses.MISSING and field.default_factory is not dataclasses.MISSING:
            value = field.default_factory()
        if isinstance(value, TiledCameraCfg):
            resized = copy.deepcopy(value)
            resized.height = int(height)
            resized.width = int(width)
            annotations[field.name] = TiledCameraCfg
            camera_attrs[field.name] = resized
            found_camera = True
    if not found_camera:
        raise ValueError(f"Camera config class {camera_cfg_cls.__name__} does not contain a TiledCameraCfg.")
    class_name = f"{camera_cfg_cls.__name__}_{int(height)}x{int(width)}"
    return configclass(type(class_name, (camera_cfg_cls,), camera_attrs))


def _resized_camera_field(camera_cfg_cls: type, field_name: str, *, height: int, width: int) -> TiledCameraCfg:
    for field in dataclasses.fields(camera_cfg_cls):
        if field.name != field_name:
            continue
        value = field.default
        if value is dataclasses.MISSING and field.default_factory is not dataclasses.MISSING:
            value = field.default_factory()
        if not isinstance(value, TiledCameraCfg):
            raise TypeError(f"{camera_cfg_cls.__name__}.{field_name} is not a TiledCameraCfg.")
        resized = copy.deepcopy(value)
        resized.height = int(height)
        resized.width = int(width)
        return resized
    raise ValueError(f"Camera config class {camera_cfg_cls.__name__} does not contain field {field_name!r}.")


def with_camera_resolution(cameras: list[type], *, height: int, width: int) -> tuple[list[type], type]:
    """Return camera preset and Droid robot config rendering at ``height`` x ``width``.

    Wrist camera resolution is owned by ``DroidCfg.wrist_cam``, while the wrist
    preset class is only used for observation introspection. Return both so
    callers can register a consistent scene.
    """
    resized_cameras = [
        _camera_cfg_class_with_resolution(camera_cfg_cls, height=height, width=width)
        for camera_cfg_cls in cameras
    ]
    resized_wrist = _resized_camera_field(WristCameraCfg, "wrist_cam", height=height, width=width)
    robot_cfg = configclass(
        type(
            f"DroidCfg_{int(height)}x{int(width)}",
            (DroidCfg,),
            {"__annotations__": {"wrist_cam": TiledCameraCfg}, "wrist_cam": resized_wrist},
        )
    )
    return resized_cameras, robot_cfg
