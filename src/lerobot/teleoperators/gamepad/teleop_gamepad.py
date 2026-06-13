# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import sys
from enum import IntEnum
from typing import Any

import numpy as np

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..teleoperator import Teleoperator
from ..utils import TeleopEvents
from .configuration_gamepad import GamepadTeleopConfig


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


gripper_action_map = {
    "close": GripperAction.CLOSE.value,
    "open": GripperAction.OPEN.value,
    "stay": GripperAction.STAY.value,
}

TCP_DELTA_ACTION_KEYS = (
    "tcp_delta_x",
    "tcp_delta_y",
    "tcp_delta_z",
    "tcp_delta_rx",
    "tcp_delta_ry",
    "tcp_delta_rz",
)


class GamepadTeleop(Teleoperator):
    """
    Teleop class to use gamepad inputs for control.
    """

    config_class = GamepadTeleopConfig
    name = "gamepad"

    def __init__(self, config: GamepadTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        if self.config.output_mode not in {"delta", "tcp_delta"}:
            raise ValueError(
                f"Unsupported gamepad output_mode '{self.config.output_mode}'. "
                "Expected one of: 'delta', 'tcp_delta'."
            )

        self.gamepad = None

    @property
    def action_features(self) -> dict:
        if self.config.output_mode == "tcp_delta":
            action_keys = TCP_DELTA_ACTION_KEYS
            if self.config.use_gripper:
                action_keys = (*TCP_DELTA_ACTION_KEYS, "gripper")
            return dict.fromkeys(action_keys, float)

        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (4,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (3,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
            }

    @property
    def feedback_features(self) -> dict:
        return {}

    def connect(self) -> None:
        # use HidApi for macos
        if sys.platform == "darwin":
            # NOTE: On macOS, pygame doesn’t reliably detect input from some controllers so we fall back to hidapi
            from .gamepad_utils import GamepadControllerHID as Gamepad
        else:
            from .gamepad_utils import GamepadController as Gamepad

        self.gamepad = Gamepad(deadzone=self.config.deadzone)
        self.gamepad.start()

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        # Update the controller to get fresh inputs
        self.gamepad.update()

        if self.config.output_mode == "tcp_delta":
            return self._get_tcp_delta_action()

        # Get movement deltas from the controller
        delta_x, delta_y, delta_z = self.gamepad.get_deltas()

        # Create action from gamepad input
        gamepad_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

        action_dict = {
            "delta_x": gamepad_action[0],
            "delta_y": gamepad_action[1],
            "delta_z": gamepad_action[2],
        }

        # Default gripper action is to stay
        gripper_action = GripperAction.STAY.value
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            gripper_action = gripper_action_map[gripper_command]
            action_dict["gripper"] = gripper_action

        return action_dict

    def _get_tcp_delta_action(self) -> RobotAction:
        if self.config.require_deadman and not self.gamepad.is_deadman_pressed():
            action_dict = {key: 0.0 for key in TCP_DELTA_ACTION_KEYS}
            if self.config.use_gripper:
                gripper_command = self.gamepad.gripper_command()
                action_dict["gripper"] = gripper_action_map[gripper_command]
            return action_dict

        x_input, y_input, z_input = self.gamepad.get_motion_inputs()
        if self.gamepad.is_rotation_mode():
            linear_delta = (0.0, 0.0, 0.0)
            angular_delta = (
                x_input * self.config.angular_delta_step_rad,
                y_input * self.config.angular_delta_step_rad,
                z_input * self.config.angular_delta_step_rad,
            )
        else:
            linear_delta = (
                x_input * self.config.linear_delta_step_m,
                y_input * self.config.linear_delta_step_m,
                z_input * self.config.linear_delta_step_m,
            )
            angular_delta = (0.0, 0.0, 0.0)

        action_dict = {
            "tcp_delta_x": float(linear_delta[0]),
            "tcp_delta_y": float(linear_delta[1]),
            "tcp_delta_z": float(linear_delta[2]),
            "tcp_delta_rx": float(angular_delta[0]),
            "tcp_delta_ry": float(angular_delta[1]),
            "tcp_delta_rz": float(angular_delta[2]),
        }
        if self.config.use_gripper:
            gripper_command = self.gamepad.gripper_command()
            action_dict["gripper"] = gripper_action_map[gripper_command]
        return action_dict

    def get_teleop_events(self) -> dict[str, Any]:
        """
        Get extra control events from the gamepad such as intervention status,
        episode termination, success indicators, etc.

        Returns:
            Dictionary containing:
                - is_intervention: bool - Whether human is currently intervening
                - terminate_episode: bool - Whether to terminate the current episode
                - success: bool - Whether the episode was successful
                - rerecord_episode: bool - Whether to rerecord the episode
        """
        if self.gamepad is None:
            return {
                TeleopEvents.IS_INTERVENTION: False,
                TeleopEvents.TERMINATE_EPISODE: False,
                TeleopEvents.SUCCESS: False,
                TeleopEvents.RERECORD_EPISODE: False,
            }

        # Update gamepad state to get fresh inputs
        self.gamepad.update()

        # Check if intervention is active
        is_intervention = self.gamepad.should_intervene()

        # Get episode end status
        episode_end_status = self.gamepad.get_episode_end_status()
        terminate_episode = episode_end_status in [
            TeleopEvents.RERECORD_EPISODE,
            TeleopEvents.FAILURE,
        ]
        success = episode_end_status == TeleopEvents.SUCCESS
        rerecord_episode = episode_end_status == TeleopEvents.RERECORD_EPISODE

        return {
            TeleopEvents.IS_INTERVENTION: is_intervention,
            TeleopEvents.TERMINATE_EPISODE: terminate_episode,
            TeleopEvents.SUCCESS: success,
            TeleopEvents.RERECORD_EPISODE: rerecord_episode,
        }

    def disconnect(self) -> None:
        """Disconnect from the gamepad."""
        if self.gamepad is not None:
            self.gamepad.stop()
            self.gamepad = None

    @property
    def is_connected(self) -> bool:
        """Check if gamepad is connected."""
        return self.gamepad is not None

    def calibrate(self) -> None:
        """Calibrate the gamepad."""
        # No calibration needed for gamepad
        pass

    @property
    def is_calibrated(self) -> bool:
        """Check if gamepad is calibrated."""
        # Gamepad doesn't require calibration
        return True

    def configure(self) -> None:
        """Configure the gamepad."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict) -> None:
        """Send feedback to the gamepad."""
        # Gamepad doesn't support feedback
        pass
