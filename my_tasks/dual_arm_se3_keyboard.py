# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard controller for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

import carb
import omni

from KUKA_Project.devices.device_base import DeviceBase


class DualArmSe3Keyboard(DeviceBase):
    """A keyboard controller for sending SE(3) commands as delta poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands.

    The command comprises of two parts:

    * delta pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle left gripper (open/close)    K
        Toggle right gripper (open/close)   H
        Move left arm along x-axis     W                 S
        Move left arm along y-axis     A                 D
        Move left arm along z-axis     Q                 E
        Rotate left arm along x-axis   Z                 X
        Rotate left arm along y-axis   T                 G
        Rotate left arm along z-axis   C                 V
        Move right arm along x-axis    I                 K
        Move right arm along y-axis    J                 L
        Move right arm along z-axis    U                 O
        Rotate right arm along x-axis  M                 N
        Rotate right arm along y-axis  P                 ;
        Rotate right arm along z-axis  ,                 .
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, pos_sensitivity: float = 0.4, rot_sensitivity: float = 0.8):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        # store inputs
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(
                event, *args
            ),
        )
        # bindings for keyboard to command
        self._create_key_bindings()
        # command buffers
        self._close_left_gripper = False
        self._close_right_gripper = False
        self._left_delta_pos = np.zeros(3)  # (x, y, z)
        self._right_delta_pos = np.zeros(3)  # (x, y, z)
        self._left_delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._right_delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"Keyboard Controller for SE(3): {self.__class__.__name__}\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tToggle left gripper (open/close): G\n"
        msg += "\tToggle right gripper (open/close): H\n"
        msg += "\tMove left arm along x-axis: W/S\n"
        msg += "\tMove left arm along y-axis: A/D\n"
        msg += "\tMove left arm along z-axis: Q/E\n"
        msg += "\tRotate left arm along x-axis: Z/X\n"
        msg += "\tRotate left arm along y-axis: R/F\n"
        msg += "\tRotate left arm along z-axis: C/V\n"
        msg += "\tMove right arm along x-axis: I/K\n"
        msg += "\tMove right arm along y-axis: J/L\n"
        msg += "\tMove right arm along z-axis: U/O\n"
        msg += "\tRotate right arm along x-axis: N/M\n"
        msg += "\tRotate right arm along y-axis: P/;\n"
        msg += "\tRotate right arm along z-axis: ,/."
        return msg

    """
    Operations
    """

    def reset(self):
        # default flags
        self._close_left_gripper = False
        self._close_right_gripper = False
        self._left_delta_pos = np.zeros(3)  # (x, y, z)
        self._right_delta_pos = np.zeros(3)  # (x, y, z)
        self._left_delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._right_delta_rot = np.zeros(3)  # (roll, pitch, yaw)

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind keyboard.

        A list of available keys are present in the
        `carb documentation <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

        Args:
            key: The keyboard button to check against.
            func: The function to call when key is pressed. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func

    def advance(self) -> tuple[np.ndarray, tuple[bool, bool]]:
        """Provides the result from keyboard event state.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # convert to rotation vectors
        left_rot_vec = Rotation.from_euler("XYZ", self._left_delta_rot).as_rotvec()
        right_rot_vec = Rotation.from_euler("XYZ", self._right_delta_rot).as_rotvec()
        # return the concatenated command and gripper states
        return np.concatenate(
            [self._left_delta_pos, left_rot_vec, self._right_delta_pos, right_rot_vec]
        ), (self._close_left_gripper, self._close_right_gripper)

    """
    Internal helpers.
    """

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Subscriber callback to when kit is updated.

        Reference:
            https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html
        """
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "B":
                self.reset()
            if event.input.name == "G":
                self._close_left_gripper = not self._close_left_gripper
            if event.input.name == "H":
                self._close_right_gripper = not self._close_right_gripper
            elif event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._left_delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "R", "F", "C", "V"]:
                self._left_delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["I", "K", "J", "L", "U", "O"]:
                self._right_delta_pos += self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["M", "N", "P", ";", ",", "."]:
                self._right_delta_rot += self._INPUT_KEY_MAPPING[event.input.name]
        # remove the command when un-pressed
        if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["W", "S", "A", "D", "Q", "E"]:
                self._left_delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["Z", "X", "R", "F", "C", "V"]:
                self._left_delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["I", "K", "J", "L", "U", "O"]:
                self._right_delta_pos -= self._INPUT_KEY_MAPPING[event.input.name]
            elif event.input.name in ["M", "N", "P", ";", ",", "."]:
                self._right_delta_rot -= self._INPUT_KEY_MAPPING[event.input.name]
        # additional callbacks
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name in self._additional_callbacks:
                self._additional_callbacks[event.input.name]()

        # since no error, we are fine :)
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._INPUT_KEY_MAPPING = {
            # toggle: gripper command
            "G": True,
            "H": True,
            # left arm x-axis (forward)
            "W": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # left arm y-axis (left-right)
            "A": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # left arm z-axis (up-down)
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # left arm roll (around x-axis)
            "Z": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "X": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # left arm pitch (around y-axis)
            "R": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            "F": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # left arm yaw (around z-axis)
            "C": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            "V": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
            # right arm x-axis (forward)
            "I": np.asarray([1.0, 0.0, 0.0]) * self.pos_sensitivity,
            "K": np.asarray([-1.0, 0.0, 0.0]) * self.pos_sensitivity,
            # right arm y-axis (left-right)
            "J": np.asarray([0.0, 1.0, 0.0]) * self.pos_sensitivity,
            "L": np.asarray([0.0, -1.0, 0.0]) * self.pos_sensitivity,
            # right arm z-axis (up-down)
            "U": np.asarray([0.0, 0.0, 1.0]) * self.pos_sensitivity,
            "O": np.asarray([0.0, 0.0, -1.0]) * self.pos_sensitivity,
            # right arm roll (around x-axis)
            "M": np.asarray([1.0, 0.0, 0.0]) * self.rot_sensitivity,
            "N": np.asarray([-1.0, 0.0, 0.0]) * self.rot_sensitivity,
            # right arm pitch (around y-axis)
            "P": np.asarray([0.0, 1.0, 0.0]) * self.rot_sensitivity,
            ";": np.asarray([0.0, -1.0, 0.0]) * self.rot_sensitivity,
            # right arm yaw (around z-axis)
            ",": np.asarray([0.0, 0.0, 1.0]) * self.rot_sensitivity,
            ".": np.asarray([0.0, 0.0, -1.0]) * self.rot_sensitivity,
        }
