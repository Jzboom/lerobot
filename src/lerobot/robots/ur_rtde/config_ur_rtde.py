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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("ur_rtde")
@dataclass
class URRTDERobotConfig(RobotConfig):
    ip: str
    control_hz: int = 30
    speed_acceleration: float = 0.15
    speed_stop_acceleration: float = 1.0
    max_linear_delta_m: float = 0.05
    max_angular_delta_rad: float = 0.05
    zero_epsilon: float = 1e-9
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    gripper_enabled: bool = False
    gripper_port: str | None = None
    gripper_baudrate: int = 115200
    gripper_slave_id: int = 1
    gripper_open_point: int = 0
    gripper_close_point: int = 1
    gripper_pulse_s: float = 0.05
    gripper_initial_pos: float = 1.0
    gripper_reset_error_on_connect: bool = False
    gripper_servo_on_connect: bool = True
    gripper_initialize_on_connect: bool = False
    gripper_setup_on_connect: bool = False
    gripper_open_pos_mm: float = 0.0
    gripper_open_speed_mm_s: float = 50.0
    gripper_close_dist_mm: float = 26.0
    gripper_close_force: float = 0.3
    gripper_close_speed_mm_s: float = 20.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.control_hz <= 0:
            raise ValueError("control_hz must be positive.")
        if self.speed_acceleration <= 0:
            raise ValueError("speed_acceleration must be positive.")
        if self.speed_stop_acceleration <= 0:
            raise ValueError("speed_stop_acceleration must be positive.")
        if self.max_linear_delta_m <= 0:
            raise ValueError("max_linear_delta_m must be positive.")
        if self.max_angular_delta_rad <= 0:
            raise ValueError("max_angular_delta_rad must be positive.")
        if self.gripper_baudrate <= 0:
            raise ValueError("gripper_baudrate must be positive.")
        if self.gripper_slave_id <= 0:
            raise ValueError("gripper_slave_id must be positive.")
        if self.gripper_open_point < 0 or self.gripper_close_point < 0:
            raise ValueError("gripper point numbers must be non-negative.")
        if self.gripper_pulse_s <= 0:
            raise ValueError("gripper_pulse_s must be positive.")
        if not 0.0 <= self.gripper_initial_pos <= 1.0:
            raise ValueError("gripper_initial_pos must be in [0, 1].")
        if self.gripper_close_force < 0:
            raise ValueError("gripper_close_force must be non-negative.")
