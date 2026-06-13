#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import require_package

from ..robot import Robot
from .config_ur_rtde import URRTDERobotConfig
from .gripper import RMGripper, find_gripper_port

if TYPE_CHECKING:
    from rtde_control import RTDEControlInterface as RTDEControlInterfaceType
    from rtde_receive import RTDEReceiveInterface as RTDEReceiveInterfaceType
else:
    RTDEControlInterfaceType = Any
    RTDEReceiveInterfaceType = Any

RTDEControlInterface = None
RTDEReceiveInterface = None

TCP_DELTA_ACTION_KEYS = (
    "tcp_delta_x",
    "tcp_delta_y",
    "tcp_delta_z",
    "tcp_delta_rx",
    "tcp_delta_ry",
    "tcp_delta_rz",
)
TCP_POSE_OBS_KEYS = (
    "tcp_pose_x",
    "tcp_pose_y",
    "tcp_pose_z",
    "tcp_pose_rx",
    "tcp_pose_ry",
    "tcp_pose_rz",
)
JOINT_OBS_KEYS = tuple(f"joint_{i}_pos" for i in range(6))
GRIPPER_ACTION_KEY = "gripper"
GRIPPER_OBS_KEYS = ("gripper_pos",)
GRIPPER_CLOSE = 0
GRIPPER_STAY = 1
GRIPPER_OPEN = 2


def _load_rtde_interfaces():
    global RTDEControlInterface, RTDEReceiveInterface
    if RTDEControlInterface is None:
        require_package("ur-rtde", extra="ur_rtde", import_name="rtde_control")
        from rtde_control import RTDEControlInterface as _RTDEControlInterface

        RTDEControlInterface = _RTDEControlInterface
    if RTDEReceiveInterface is None:
        require_package("ur-rtde", extra="ur_rtde", import_name="rtde_receive")
        from rtde_receive import RTDEReceiveInterface as _RTDEReceiveInterface

        RTDEReceiveInterface = _RTDEReceiveInterface
    return RTDEControlInterface, RTDEReceiveInterface


class URRTDERobot(Robot):
    """Universal Robots adapter using ur_rtde Cartesian speed control."""

    config_class = URRTDERobotConfig
    name = "ur_rtde"

    def __init__(self, config: URRTDERobotConfig):
        super().__init__(config)
        self.config = config
        self.cameras = make_cameras_from_configs(config.cameras)
        self.rtde_c: RTDEControlInterfaceType | None = None
        self.rtde_r: RTDEReceiveInterfaceType | None = None
        self.gripper: RMGripper | None = None
        self._last_gripper_command = GRIPPER_STAY
        self._gripper_estimated_pos = config.gripper_initial_pos
        self._pending_gripper_point: int | None = None
        self._is_connected = False
        self.logs: dict[str, float] = {}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        state_features = dict.fromkeys((*TCP_POSE_OBS_KEYS, *JOINT_OBS_KEYS), float)
        if self.config.gripper_enabled:
            state_features.update(dict.fromkeys(GRIPPER_OBS_KEYS, float))
        camera_features = {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }
        return {**state_features, **camera_features}

    @cached_property
    def action_features(self) -> dict[str, type]:
        action_keys = TCP_DELTA_ACTION_KEYS
        if self.config.gripper_enabled:
            action_keys = (*TCP_DELTA_ACTION_KEYS, GRIPPER_ACTION_KEY)
        return dict.fromkeys(action_keys, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        rtde_control_cls, rtde_receive_cls = _load_rtde_interfaces()
        self.rtde_c = rtde_control_cls(self.config.ip)
        self.rtde_r = rtde_receive_cls(self.config.ip)
        if self.config.gripper_enabled:
            gripper_port = self.config.gripper_port or find_gripper_port()
            if gripper_port is None:
                raise ConnectionError(
                    "gripper_enabled is true but no gripper serial port was found. "
                    "Set --robot.gripper_port=/dev/ttyUSBx or disable the gripper."
                )
            self.gripper = RMGripper(
                port=gripper_port,
                baudrate=self.config.gripper_baudrate,
                slave_id=self.config.gripper_slave_id,
                pulse_s=self.config.gripper_pulse_s,
            )
            self.gripper.connect()
            if self.config.gripper_reset_error_on_connect:
                self.gripper.reset_error()
            if self.config.gripper_servo_on_connect:
                self.gripper.servo_on()
            if self.config.gripper_initialize_on_connect:
                self.gripper.initialize()
            if self.config.gripper_setup_on_connect:
                self.gripper.setup_point_absolute(
                    point_num=self.config.gripper_open_point,
                    target_pos=self.config.gripper_open_pos_mm,
                    speed=self.config.gripper_open_speed_mm_s,
                )
                self.gripper.setup_point_push(
                    point_num=self.config.gripper_close_point,
                    dist=self.config.gripper_close_dist_mm,
                    force_percent=self.config.gripper_close_force,
                    speed=self.config.gripper_close_speed_mm_s,
                )
        for cam in self.cameras.values():
            cam.connect()
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self.rtde_r is None:
            raise ConnectionError("RTDE receive interface is not connected.")

        before_read_t = time.perf_counter()
        tcp_pose = self.rtde_r.getActualTCPPose()
        joints = self.rtde_r.getActualQ()
        self.logs["read_state_dt_s"] = time.perf_counter() - before_read_t

        obs: RobotObservation = {
            key: float(value) for key, value in zip(TCP_POSE_OBS_KEYS, tcp_pose, strict=True)
        }
        obs.update({key: float(value) for key, value in zip(JOINT_OBS_KEYS, joints, strict=True)})
        if self.config.gripper_enabled:
            obs.update(self._get_gripper_observation())

        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        if self.rtde_c is None:
            raise ConnectionError("RTDE control interface is not connected.")

        sent_action = self._clip_action(action)
        if self.config.gripper_enabled:
            self._send_gripper_action(int(sent_action[GRIPPER_ACTION_KEY]))
        delta = np.array([sent_action[key] for key in TCP_DELTA_ACTION_KEYS], dtype=np.float64)

        if np.linalg.norm(delta) <= self.config.zero_epsilon:
            self.rtde_c.speedStop(self.config.speed_stop_acceleration)
            return sent_action

        control_dt = 1.0 / self.config.control_hz
        speed = (delta / control_dt).tolist()
        before_write_t = time.perf_counter()
        self.rtde_c.speedL(speed, self.config.speed_acceleration, control_dt)
        self.logs["write_speed_dt_s"] = time.perf_counter() - before_write_t
        return sent_action

    def _clip_action(self, action: RobotAction) -> RobotAction:
        missing = [key for key in TCP_DELTA_ACTION_KEYS if key not in action]
        if self.config.gripper_enabled and GRIPPER_ACTION_KEY not in action:
            missing.append(GRIPPER_ACTION_KEY)
        if missing:
            raise KeyError(f"Missing UR RTDE action keys: {missing}")

        sent_action: RobotAction = {}
        for key in TCP_DELTA_ACTION_KEYS[:3]:
            sent_action[key] = float(
                np.clip(action[key], -self.config.max_linear_delta_m, self.config.max_linear_delta_m)
            )
        for key in TCP_DELTA_ACTION_KEYS[3:]:
            sent_action[key] = float(
                np.clip(action[key], -self.config.max_angular_delta_rad, self.config.max_angular_delta_rad)
            )
        if self.config.gripper_enabled:
            sent_action[GRIPPER_ACTION_KEY] = float(
                np.clip(round(float(action[GRIPPER_ACTION_KEY])), GRIPPER_CLOSE, GRIPPER_OPEN)
            )
        return sent_action

    def _send_gripper_action(self, gripper_command: int) -> None:
        if self.gripper is None:
            raise ConnectionError("Gripper is enabled but not connected.")
        if gripper_command == self._last_gripper_command:
            return

        self._last_gripper_command = gripper_command
        if gripper_command == GRIPPER_STAY:
            return

        if gripper_command == GRIPPER_OPEN:
            self.gripper.trigger_point(self.config.gripper_open_point)
            self._pending_gripper_point = self.config.gripper_open_point
        elif gripper_command == GRIPPER_CLOSE:
            self.gripper.trigger_point(self.config.gripper_close_point)
            self._pending_gripper_point = self.config.gripper_close_point

    def _get_gripper_observation(self) -> RobotObservation:
        if self.gripper is None:
            raise ConnectionError("Gripper is enabled but not connected.")

        open_done = self.gripper.is_point_done(self.config.gripper_open_point)
        close_done = self.gripper.is_point_done(self.config.gripper_close_point)

        if self._pending_gripper_point == self.config.gripper_open_point and open_done:
            self._gripper_estimated_pos = 1.0
            self._pending_gripper_point = None
        elif self._pending_gripper_point == self.config.gripper_close_point and close_done:
            self._gripper_estimated_pos = 0.0
            self._pending_gripper_point = None
        elif self._pending_gripper_point is None:
            if open_done and not close_done:
                self._gripper_estimated_pos = 1.0
            elif close_done and not open_done:
                self._gripper_estimated_pos = 0.0

        return {
            "gripper_pos": float(self._gripper_estimated_pos),
        }

    def disconnect(self) -> None:
        if self.rtde_c is not None:
            try:
                self.rtde_c.speedStop(self.config.speed_stop_acceleration)
            finally:
                self.rtde_c.stopScript()
                if hasattr(self.rtde_c, "disconnect"):
                    self.rtde_c.disconnect()
                self.rtde_c = None

        if self.rtde_r is not None:
            if hasattr(self.rtde_r, "disconnect"):
                self.rtde_r.disconnect()
            self.rtde_r = None

        if self.gripper is not None:
            self.gripper.disconnect()
            self.gripper = None

        for cam in self.cameras.values():
            cam.disconnect()

        self._is_connected = False
