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

from lerobot.teleoperators.gamepad import GamepadTeleop, GamepadTeleopConfig
from lerobot.teleoperators.gamepad.teleop_gamepad import TCP_DELTA_ACTION_KEYS


class FakeGamepad:
    def __init__(self, motion=(0.0, 0.0, 0.0), deadman=False, rotation=False, gripper="stay"):
        self.motion = motion
        self.deadman = deadman
        self.rotation = rotation
        self.gripper = gripper
        self.update_count = 0

    def update(self):
        self.update_count += 1

    def is_deadman_pressed(self):
        return self.deadman

    def is_rotation_mode(self):
        return self.rotation

    def get_motion_inputs(self):
        return self.motion

    def gripper_command(self):
        return self.gripper


def _make_tcp_delta_teleop(gamepad, use_gripper=False):
    teleop = GamepadTeleop(
        GamepadTeleopConfig(
            output_mode="tcp_delta",
            use_gripper=use_gripper,
            require_deadman=True,
            linear_delta_step_m=0.001,
            angular_delta_step_rad=0.01,
        )
    )
    teleop.gamepad = gamepad
    return teleop


def test_tcp_delta_mode_outputs_zero_without_deadman():
    teleop = _make_tcp_delta_teleop(FakeGamepad(motion=(1.0, -0.5, 0.25), deadman=False))

    action = teleop.get_action()

    assert action == {key: 0.0 for key in TCP_DELTA_ACTION_KEYS}


def test_tcp_delta_mode_maps_translation_when_deadman_is_held():
    teleop = _make_tcp_delta_teleop(FakeGamepad(motion=(1.0, -0.5, 0.25), deadman=True))

    action = teleop.get_action()

    assert action == {
        "tcp_delta_x": 0.001,
        "tcp_delta_y": -0.0005,
        "tcp_delta_z": 0.00025,
        "tcp_delta_rx": 0.0,
        "tcp_delta_ry": 0.0,
        "tcp_delta_rz": 0.0,
    }


def test_tcp_delta_mode_maps_rotation_when_lb_and_rb_are_held():
    teleop = _make_tcp_delta_teleop(
        FakeGamepad(motion=(1.0, -0.5, 0.25), deadman=True, rotation=True)
    )

    action = teleop.get_action()

    assert action == {
        "tcp_delta_x": 0.0,
        "tcp_delta_y": 0.0,
        "tcp_delta_z": 0.0,
        "tcp_delta_rx": 0.01,
        "tcp_delta_ry": -0.005,
        "tcp_delta_rz": 0.0025,
    }


def test_tcp_delta_mode_keeps_gripper_available_without_deadman():
    teleop = _make_tcp_delta_teleop(
        FakeGamepad(motion=(1.0, -0.5, 0.25), deadman=False, gripper="close"),
        use_gripper=True,
    )

    action = teleop.get_action()

    assert action == {
        "tcp_delta_x": 0.0,
        "tcp_delta_y": 0.0,
        "tcp_delta_z": 0.0,
        "tcp_delta_rx": 0.0,
        "tcp_delta_ry": 0.0,
        "tcp_delta_rz": 0.0,
        "gripper": 0,
    }
