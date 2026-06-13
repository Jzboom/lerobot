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

import logging
import inspect
import struct
import time
from typing import Any

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)


def _load_modbus_serial_client():
    require_package("pymodbus", extra="ur_rtde", import_name="pymodbus")
    try:
        from pymodbus.client import ModbusSerialClient
    except ImportError:
        from pymodbus.client.sync import ModbusSerialClient

    return ModbusSerialClient


def find_gripper_port() -> str | None:
    require_package("pyserial", extra="ur_rtde", import_name="serial")
    import serial.tools.list_ports

    usb_ports = []
    for port in serial.tools.list_ports.comports():
        if any(token in port.device for token in ("USB", "ACM", "COM")):
            logger.info("Found gripper candidate port: %s - %s", port.device, port.description)
            usb_ports.append(port)

    if not usb_ports:
        return None
    return usb_ports[0].device


class RMGripper:
    """Modbus RTU point-mode controller for the RM gripper used with the UR RTDE adapter."""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        slave_id: int = 1,
        pulse_s: float = 0.05,
        timeout_s: float = 1.0,
    ):
        self.port = port
        self.baudrate = baudrate
        self.slave_id = slave_id
        self.pulse_s = pulse_s
        self.timeout_s = timeout_s
        self.client: Any | None = None

    def connect(self) -> None:
        modbus_serial_client = _load_modbus_serial_client()
        self.client = modbus_serial_client(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=self.timeout_s,
        )
        if not self.client.connect():
            self.client = None
            raise ConnectionError(f"Failed to connect gripper on port {self.port}.")
        logger.info("Connected gripper on port %s", self.port)

    def disconnect(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
            logger.info("Disconnected gripper.")

    def reset_error(self) -> None:
        self.trigger_coil(0)

    def servo_on(self) -> None:
        self._write_coil(1, True)
        time.sleep(0.5)

    def initialize(self) -> None:
        self.trigger_coil(17)
        while True:
            time.sleep(0.5)
            if self.read_discrete_input(1037):
                return

    def trigger_point(self, point_num: int) -> None:
        self.trigger_coil(1000 + point_num)

    def is_point_done(self, point_num: int) -> bool:
        return self.read_discrete_input(1000 + point_num)

    def trigger_coil(self, address: int) -> None:
        self._write_coil(address, False)
        time.sleep(self.pulse_s)
        self._write_coil(address, True)
        time.sleep(self.pulse_s)

    def setup_point_absolute(
        self,
        point_num: int,
        target_pos: float,
        speed: float = 80.0,
        acc: float = 150.0,
        dec: float = 150.0,
        pos_range: float = 0.1,
    ) -> None:
        base_addr = 5000 + point_num * 16
        regs = []
        regs.extend(self._int_to_regs(3))
        regs.extend(self._int_to_regs(-1))
        regs.extend(self._float_to_regs(target_pos))
        regs.extend(self._float_to_regs(speed))
        regs.extend(self._float_to_regs(acc))
        regs.extend(self._float_to_regs(dec))
        regs.extend(self._float_to_regs(pos_range))
        regs.extend(self._float_to_regs(0.0))
        self._write_registers(base_addr, regs)

    def setup_point_push(
        self,
        point_num: int,
        dist: float,
        force_percent: float,
        speed: float = 20.0,
        acc: float = 100.0,
        pos_range: float = 0.1,
        time_range: float = 100.0,
    ) -> None:
        base_addr = 5000 + point_num * 16
        regs = []
        regs.extend(self._int_to_regs(4))
        regs.extend(self._int_to_regs(-1))
        regs.extend(self._float_to_regs(dist))
        regs.extend(self._float_to_regs(speed))
        regs.extend(self._float_to_regs(acc))
        regs.extend(self._float_to_regs(force_percent))
        regs.extend(self._float_to_regs(pos_range))
        regs.extend(self._float_to_regs(time_range))
        self._write_registers(base_addr, regs)

    def _write_coil(self, address: int, value: bool) -> None:
        result = self._call_with_slave("write_coil", address, value)
        self._raise_if_error(result, f"write coil {address}")

    def _write_registers(self, address: int, values: list[int]) -> None:
        result = self._call_with_slave("write_registers", address, values)
        self._raise_if_error(result, f"write registers {address}")

    def read_discrete_input(self, address: int) -> bool:
        result = self._call_with_slave("read_discrete_inputs", address, count=1)
        self._raise_if_error(result, f"read discrete input {address}")
        return bool(result.bits[0])

    def _call_with_slave(self, method_name: str, *args, **kwargs):
        if self.client is None:
            raise ConnectionError("Gripper is not connected.")

        method = getattr(self.client, method_name)
        signature = inspect.signature(method)
        for slave_kwarg in ("device_id", "slave", "unit"):
            if slave_kwarg in signature.parameters:
                return method(*args, **{slave_kwarg: self.slave_id}, **kwargs)
        return method(*args, **kwargs)

    @staticmethod
    def _raise_if_error(result, context: str) -> None:
        if result is None:
            raise ConnectionError(f"Gripper Modbus {context} returned no result.")
        if hasattr(result, "isError") and result.isError():
            raise ConnectionError(f"Gripper Modbus {context} failed: {result}")

    @staticmethod
    def _float_to_regs(value: float) -> list[int]:
        data = struct.pack(">f", value)
        return [(data[2] << 8) | data[3], (data[0] << 8) | data[1]]

    @staticmethod
    def _int_to_regs(value: int) -> list[int]:
        data = struct.pack(">i", value)
        return [(data[2] << 8) | data[3], (data[0] << 8) | data[1]]
