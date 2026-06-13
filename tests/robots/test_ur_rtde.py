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

from unittest.mock import MagicMock

import pytest

from lerobot.datasets import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.processor import make_default_teleop_action_processor
from lerobot.robots.ur_rtde import (
    GRIPPER_ACTION_KEY,
    GRIPPER_OBS_KEYS,
    JOINT_OBS_KEYS,
    TCP_DELTA_ACTION_KEYS,
    TCP_POSE_OBS_KEYS,
    URRTDERobot,
    URRTDERobotConfig,
)
from lerobot.robots.ur_rtde import ur_rtde as ur_rtde_module


def _make_robot(monkeypatch, gripper=None, **config_overrides):
    control = MagicMock()
    receive = MagicMock()
    receive.getActualTCPPose.return_value = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03]
    receive.getActualQ.return_value = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    monkeypatch.setattr(ur_rtde_module, "RTDEControlInterface", MagicMock(return_value=control))
    monkeypatch.setattr(ur_rtde_module, "RTDEReceiveInterface", MagicMock(return_value=receive))
    if config_overrides.get("gripper_enabled"):
        if gripper is None:
            gripper = MagicMock()
            gripper.is_point_done.return_value = False
        monkeypatch.setattr(ur_rtde_module, "find_gripper_port", MagicMock(return_value="/dev/ttyUSB0"))
        monkeypatch.setattr(ur_rtde_module, "RMGripper", MagicMock(return_value=gripper))

    robot = URRTDERobot(URRTDERobotConfig(ip="192.168.1.10", control_hz=30, **config_overrides))
    robot.connect()
    return robot, control, receive


def test_tcp_delta_action_converts_to_speedl(monkeypatch):
    robot, control, _receive = _make_robot(monkeypatch)

    action = {
        "tcp_delta_x": 0.001,
        "tcp_delta_y": -0.002,
        "tcp_delta_z": 0.0,
        "tcp_delta_rx": 0.01,
        "tcp_delta_ry": 0.0,
        "tcp_delta_rz": -0.01,
    }

    sent = robot.send_action(action)

    assert sent == action
    control.speedL.assert_called_once()
    speed, acceleration, duration = control.speedL.call_args.args
    assert speed == pytest.approx([0.03, -0.06, 0.0, 0.3, 0.0, -0.3])
    assert acceleration == 0.15
    assert duration == pytest.approx(1 / 30)


def test_zero_tcp_delta_stops_speed(monkeypatch):
    robot, control, _receive = _make_robot(monkeypatch)

    robot.send_action(dict.fromkeys(TCP_DELTA_ACTION_KEYS, 0.0))

    control.speedStop.assert_called_once_with(1.0)
    control.speedL.assert_not_called()


def test_observation_contains_tcp_pose_and_joints(monkeypatch):
    robot, _control, _receive = _make_robot(monkeypatch)

    obs = robot.get_observation()

    assert list(robot.observation_features) == [*TCP_POSE_OBS_KEYS, *JOINT_OBS_KEYS]
    assert obs["tcp_pose_x"] == 0.1
    assert obs["tcp_pose_rz"] == 0.03
    assert obs["joint_0_pos"] == 0.0
    assert obs["joint_5_pos"] == 0.5


def test_dataset_action_features_are_tcp_deltas(monkeypatch):
    robot, _control, _receive = _make_robot(monkeypatch)

    features = aggregate_pipeline_dataset_features(
        pipeline=make_default_teleop_action_processor(),
        initial_features=create_initial_features(action=robot.action_features),
    )

    assert features["action"]["names"] == list(TCP_DELTA_ACTION_KEYS)


def test_gripper_action_triggers_point_once(monkeypatch):
    gripper = MagicMock()
    gripper.is_point_done.return_value = False
    robot, _control, _receive = _make_robot(monkeypatch, gripper=gripper, gripper_enabled=True)

    action = dict.fromkeys(TCP_DELTA_ACTION_KEYS, 0.0)
    action[GRIPPER_ACTION_KEY] = 0
    robot.send_action(action)
    robot.send_action(action)

    gripper.trigger_point.assert_called_once_with(1)

    action[GRIPPER_ACTION_KEY] = 1
    robot.send_action(action)
    action[GRIPPER_ACTION_KEY] = 2
    robot.send_action(action)

    assert gripper.trigger_point.call_args_list[-1].args == (0,)


def test_gripper_observation_contains_estimated_position(monkeypatch):
    gripper = MagicMock()
    gripper.is_point_done.side_effect = lambda point: point == 0
    robot, _control, _receive = _make_robot(monkeypatch, gripper=gripper, gripper_enabled=True)

    obs = robot.get_observation()

    assert list(robot.observation_features) == [*TCP_POSE_OBS_KEYS, *JOINT_OBS_KEYS, *GRIPPER_OBS_KEYS]
    assert obs["gripper_pos"] == 1.0


def test_dataset_action_features_include_gripper_when_enabled(monkeypatch):
    robot, _control, _receive = _make_robot(monkeypatch, gripper_enabled=True)

    features = aggregate_pipeline_dataset_features(
        pipeline=make_default_teleop_action_processor(),
        initial_features=create_initial_features(action=robot.action_features),
    )

    assert features["action"]["names"] == [*TCP_DELTA_ACTION_KEYS, GRIPPER_ACTION_KEY]
