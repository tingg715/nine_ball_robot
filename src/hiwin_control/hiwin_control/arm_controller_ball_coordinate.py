#!/usr/bin/env python3
"""
Hiwin Base-Coordinate Position Test UI

Input:
  /ball_coordinate
  type: std_msgs/msg/String

Expected JSON:
  [
    {"label":"1", "x":120.5, "y":480.2, "z":-120.7},
    {"label":"white", "x":350.8, "y":450.3, "z":-120.7}
  ]

Behavior:
  1. Move the robot to arm.yaml -> armpos.
  2. Receive each ball's already-converted Base X/Y/Z.
  3. Map Base X/Y to the table UI.
  4. Let the operator freeze and select a ball.
  5. Move CUE_TOOL to the selected Base X/Y and to:
         received Base Z + TEST_ABOVE_OFFSET_MM
  6. Stop above the selected ball. No descent and no hit.

Run:
  ros2 run hiwin_control arm_controller_ball_coordinate
"""

import json
import math
import os
import signal
import threading
import time
from enum import Enum
from threading import Event, Thread

import rclpy
import tkinter as tk
import yaml
from geometry_msgs.msg import Twist
from hiwin_interfaces.srv import RobotCommand
from rclpy.node import Node
from std_msgs.msg import String
from tkinter import messagebox, ttk


# ═══════════════════════════════════════════════════
#  Hardware and safety constants
# ═══════════════════════════════════════════════════
CUE_TOOL = 8
DEFAULT_VELOCITY = 70
DEFAULT_ACCEL = 70

# The command Z is:
#   received ball Base Z + TEST_ABOVE_OFFSET_MM
#
# Keep this positive for the first tests. Set to 0 only after the
# Base coordinates and TCP/tool definition have been physically verified.
TEST_ABOVE_OFFSET_MM = 100.0

# Fixed Base-coordinate compensation. These offsets are applied exactly once,
# immediately after each /ball_coordinate message has been validated.
BALL_X_COMPENSATION_MM = 10.0
BALL_Y_COMPENSATION_MM = 0.0
BALL_Z_COMPENSATION_MM = 0.0

# Tool Rz is a Cartesian tool orientation.
# It is not the direct J6 joint angle.
TOOL_RZ_OFFSET_DEG = -90.0

# TODO: Confirm these two fixed values with the teach pendant before using
# them. The cue must remain in a safe, reasonable posture. For now, the
# movement uses Tool 8 Rx/Ry read by CHECK_POSE instead of these placeholders;
# Rx and Ry are not automatically determined from the shot direction.
CUE_RX_DEG = None
CUE_RY_DEG = None

# Safe test movement settings.
TEST_VELOCITY = 20
TEST_ACCELERATION = 20


# ═══════════════════════════════════════════════════
#  Load arm.yaml
# ═══════════════════════════════════════════════════
_CWD = os.getcwd()
_ARM_YAML_PATH = os.path.join(
    _CWD,
    'src/hiwin_control/hiwin_control/arm.yaml',
)

if not os.path.isfile(_ARM_YAML_PATH):
    raise RuntimeError(f'arm.yaml does not exist: {_ARM_YAML_PATH}')

with open(_ARM_YAML_PATH, 'r', encoding='utf-8') as _arm_file:
    _arm = yaml.safe_load(_arm_file)

if not isinstance(_arm, dict):
    raise RuntimeError(f'Invalid arm.yaml content: {_ARM_YAML_PATH}')

for _field in ('armpos', 'pot0', 'pot1', 'pot2', 'pot3'):
    if _field not in _arm:
        raise RuntimeError(
            f'arm.yaml is missing required field: {_field}'
        )

FIX_ABS_CAM = [float(value) for value in _arm['armpos']]
if len(FIX_ABS_CAM) != 6:
    raise RuntimeError(
        'arm.yaml armpos must be [x, y, z, rx, ry, rz]'
    )

SAFE_TRANSIT_X = FIX_ABS_CAM[0]
SAFE_TRANSIT_Y = FIX_ABS_CAM[1]
SAFE_TRANSIT_Z = FIX_ABS_CAM[2] + 100.0

_ARM_POTS = [
    [float(_arm['pot0'][0]), float(_arm['pot0'][1])],
    [float(_arm['pot1'][0]), float(_arm['pot1'][1])],
    [float(_arm['pot2'][0]), float(_arm['pot2'][1])],
    [float(_arm['pot3'][0]), float(_arm['pot3'][1])],
]

_TBL_X0 = min(point[0] for point in _ARM_POTS)
_TBL_X1 = max(point[0] for point in _ARM_POTS)
_TBL_Y0 = min(point[1] for point in _ARM_POTS)
_TBL_Y1 = max(point[1] for point in _ARM_POTS)

if _TBL_X1 == _TBL_X0 or _TBL_Y1 == _TBL_Y0:
    raise RuntimeError('Invalid table Base-coordinate bounds in arm.yaml')


# ═══════════════════════════════════════════════════
#  UI layout constants
# ═══════════════════════════════════════════════════
TABLE_WIDTH_MM = 627
TABLE_HEIGHT_MM = 304

UPAD = 36
UTW = TABLE_WIDTH_MM
UTH = TABLE_HEIGHT_MM
UW = UTW + UPAD * 2
UH = UTH + UPAD * 2

BR = 16
PR = 20


def arm_mm_to_canvas(arm_x, arm_y):
    """Robot Base X/Y in mm -> Tkinter canvas X/Y."""
    canvas_x = (
        UPAD
        + (float(arm_x) - _TBL_X0)
        / (_TBL_X1 - _TBL_X0)
        * UTW
    )
    canvas_y = (
        UPAD
        + (_TBL_Y1 - float(arm_y))
        / (_TBL_Y1 - _TBL_Y0)
        * UTH
    )
    return canvas_x, canvas_y


def normalize_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0
    while angle <= -180.0:
        angle += 360.0
    return angle


# Pocket canvas positions derived from Base coordinates in arm.yaml.
_pc = [arm_mm_to_canvas(x, y) for x, y in _ARM_POTS]
_mx = (_ARM_POTS[0][0] + _ARM_POTS[1][0]) / 2.0
_mty = (_ARM_POTS[0][1] + _ARM_POTS[1][1]) / 2.0
_mby = (_ARM_POTS[2][1] + _ARM_POTS[3][1]) / 2.0

POCKET_POSITIONS = [
    (*_pc[0], 'TL', 'Top-Left'),
    (*arm_mm_to_canvas(_mx, _mty), 'TM', 'Top-Mid'),
    (*_pc[1], 'TR', 'Top-Right'),
    (*_pc[3], 'BL', 'Bot-Left'),
    (*arm_mm_to_canvas(_mx, _mby), 'BM', 'Bot-Mid'),
    (*_pc[2], 'BR', 'Bot-Right'),
]

BALL_COLORS = {
    'white': '#ffffff',
    'cue': '#ffffff',
    'w': '#ffffff',
    '1': '#f5e642',
    '2': '#1a3ab5',
    '3': '#e74c3c',
    '4': '#6e2da8',
    '5': '#e67e22',
    '6': '#1a7a3a',
    '7': '#8b1a1a',
    '8': '#222222',
    '9': '#f5e642',
}

UI_WATCH = 'WATCHING'
UI_SELECT = 'SELECT_TARGET'
UI_PREVIEW = 'PREVIEW'
UI_HITTING = 'HITTING'


# ═══════════════════════════════════════════════════
#  Arm states
# ═══════════════════════════════════════════════════
class States(Enum):
    INIT = 0
    FINISH = 1
    MOVE_TO_PHOTO = 2
    WAIT_FOR_UI = 3
    TEST_MOVE_ABOVE = 4


# ═══════════════════════════════════════════════════
#  UI physics helpers
# ═══════════════════════════════════════════════════
def ui_simulate_path(sx, sy, vx, vy, steps, bounce=True):
    path = [(sx, sy)]
    x, y = sx, sy

    for _ in range(steps):
        x += vx
        y += vy

        if bounce:
            if x <= UPAD + BR:
                vx = abs(vx)
                x = UPAD + BR
            if x >= UW - UPAD - BR:
                vx = -abs(vx)
                x = UW - UPAD - BR
            if y <= UPAD + BR:
                vy = abs(vy)
                y = UPAD + BR
            if y >= UH - UPAD - BR:
                vy = -abs(vy)
                y = UH - UPAD - BR
        else:
            if not (
                UPAD <= x <= UW - UPAD
                and UPAD <= y <= UH - UPAD
            ):
                break

        path.append((x, y))

    return path


def ui_collision(wvx, wvy, hx, hy, tx, ty):
    dx = tx - hx
    dy = ty - hy
    distance = math.hypot(dx, dy)

    if distance == 0:
        return (wvx, wvy), (0.0, 0.0)

    nx = dx / distance
    ny = dy / distance
    dot = wvx * nx + wvy * ny
    target_vx = dot * nx
    target_vy = dot * ny

    return (
        (target_vx, target_vy),
        (wvx - target_vx, wvy - target_vy),
    )


def ui_calc_hit_angle(wx, wy, tx, ty, px, py):
    target_to_pocket = math.atan2(py - ty, px - tx)
    ghost_x = tx - math.cos(target_to_pocket) * BR * 2
    ghost_y = ty - math.sin(target_to_pocket) * BR * 2
    angle = math.degrees(
        math.atan2(ghost_y - wy, ghost_x - wx)
    ) % 360

    return angle, ghost_x, ghost_y


# ═══════════════════════════════════════════════════
#  ROS2 node
# ═══════════════════════════════════════════════════
class Hiwin_Controller(Node):

    def __init__(self):
        super().__init__('hiwin_ball_position_test')

        self.hiwin_client = self.create_client(
            RobotCommand,
            'hiwinmodbus_service',
        )

        self.ball_coordinate_subscription = self.create_subscription(
            String,
            '/ball_coordinate',
            self._ball_coordinate_cb,
            10,
        )

        self._lock = threading.Lock()
        self.all_balls_base = []
        self._coordinate_msg_count = 0
        self._printed_ball_coordinates = False

        self.current_pose = [0.0] * 6
        self.test_target_pose = None

        self._shot_event = Event()
        self._reset_event = Event()
        self.photo_pose_ready = Event()
        self.startup_failed = Event()

        # [target_x, target_y, target_z, white_x, white_y, white_z, force]
        self.ui_shot_info = None

        self.get_logger().info('Hiwin Base-coordinate position test ready')
        self.get_logger().info(
            'Subscribing: /ball_coordinate (std_msgs/msg/String)'
        )

    # ── /ball_coordinate callback ─────────────────
    def _ball_coordinate_cb(self, msg):
        try:
            decoded = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().warning(
                f'Invalid /ball_coordinate JSON: {error}'
            )
            return

        if not isinstance(decoded, list):
            self.get_logger().warning(
                '/ball_coordinate JSON must be a list.'
            )
            return

        valid_balls = []

        for index, item in enumerate(decoded):
            if not isinstance(item, dict):
                self.get_logger().warning(
                    f'Ball item {index} is not a JSON object.'
                )
                continue

            missing = [
                key
                for key in ('label', 'x', 'y', 'z')
                if key not in item
            ]
            if missing:
                self.get_logger().warning(
                    f'Ball item {index} is missing: {missing}'
                )
                continue

            try:
                label = str(item['label'])
                raw_x = float(item['x'])
                raw_y = float(item['y'])
                raw_z = float(item['z'])
            except (TypeError, ValueError) as error:
                self.get_logger().warning(
                    f'Invalid ball item {index}: {error}'
                )
                continue

            if not all(
                math.isfinite(value)
                for value in (raw_x, raw_y, raw_z)
            ):
                self.get_logger().warning(
                    f'Ball item {index} contains a non-finite coordinate.'
                )
                continue

            compensated_x = raw_x + BALL_X_COMPENSATION_MM
            compensated_y = raw_y + BALL_Y_COMPENSATION_MM
            compensated_z = raw_z + BALL_Z_COMPENSATION_MM

            valid_balls.append({
                'label': label,
                'x': compensated_x,
                'y': compensated_y,
                'z': compensated_z,
                'raw_x': raw_x,
                'raw_y': raw_y,
                'raw_z': raw_z,
            })

            self.get_logger().info(
                f'Ball {index + 1}:\n'
                f'raw Base=({raw_x:.2f}, {raw_y:.2f}, {raw_z:.2f}) mm\n'
                f'compensated Base=('
                f'{compensated_x:.2f}, '
                f'{compensated_y:.2f}, '
                f'{compensated_z:.2f}) mm'
            )

        if not valid_balls:
            return

        with self._lock:
            self.all_balls_base = valid_balls
            self._coordinate_msg_count += 1

            message_count = self._coordinate_msg_count

        if message_count == 1 or message_count % 100 == 0:
            self.get_logger().info(
                f'Base coordinates received: {len(valid_balls)} balls'
            )

    # ── UI handoff ────────────────────────────────
    def request_photo_pose(self):
        self._reset_event.set()
        self.get_logger().info(
            'Reset requested: returning to photo pose'
        )

    def set_shot_from_ui(
        self,
        target_x,
        target_y,
        target_z,
        white_x,
        white_y,
        white_z,
        force,
    ):
        with self._lock:
            self.ui_shot_info = [
                float(target_x),
                float(target_y),
                float(target_z),
                float(white_x),
                float(white_y),
                float(white_z),
                int(force),
            ]

        self._shot_event.set()
        self.get_logger().info(
            'Shot from UI: '
            f'target Base=({target_x:.2f}, '
            f'{target_y:.2f}, {target_z:.2f}), '
            f'white Base=({white_x:.2f}, '
            f'{white_y:.2f}, {white_z:.2f}), '
            f'force={force}'
        )

    def get_current_pose(self):
        with self._lock:
            return list(self.current_pose)

    def get_balls_for_ui(self):
        with self._lock:
            balls_base = [
                dict(ball)
                for ball in self.all_balls_base
            ]

        balls = []

        for ball in balls_base:
            base_x = float(ball['x'])
            base_y = float(ball['y'])
            base_z = float(ball['z'])
            canvas_x, canvas_y = arm_mm_to_canvas(
                ball['x'],
                ball['y'],
            )

            balls.append({
                'label': str(ball['label']),
                'x': canvas_x,
                'y': canvas_y,
                'base_x': base_x,
                'base_y': base_y,
                'base_z': base_z,
                'raw_x': float(ball['raw_x']),
                'raw_y': float(ball['raw_y']),
                'raw_z': float(ball['raw_z']),
            })

        if balls and not self._printed_ball_coordinates:
            for ball in balls:
                self.get_logger().info(
                    f'Ball {ball["label"]}: '
                    f'Base=({ball["base_x"]:.2f}, '
                    f'{ball["base_y"]:.2f}, '
                    f'{ball["base_z"]:.2f}), '
                    f'Canvas=({ball["x"]:.2f}, {ball["y"]:.2f})'
                )

            self._printed_ball_coordinates = True

        return balls

    # ── Arm state machine ─────────────────────────
    def _sm(self, state):
        if state == States.INIT:
            self.get_logger().info(
                'Starting directly with arm.yaml photo pose'
            )
            return States.MOVE_TO_PHOTO

        if state == States.MOVE_TO_PHOTO:
            self._printed_ball_coordinates = False
            self.test_target_pose = None

            pose = self._twist(*FIX_ABS_CAM)
            self.get_logger().info(
                f'Moving directly to photo pose: {FIX_ABS_CAM}'
            )

            response = self._call(
                self._req(
                    cmd_mode=RobotCommand.Request.PTP,
                    pose=pose,
                )
            )

            if (
                response is None
                or response.arm_state != RobotCommand.Response.IDLE
            ):
                self.get_logger().error(
                    'Failed to reach photo pose; UI will not open'
                )
                self.startup_failed.set()
                return None

            with self._lock:
                self.current_pose = list(response.current_position)

            self.photo_pose_ready.set()
            self.get_logger().info(
                'Photo pose reached; opening UI'
            )
            return States.WAIT_FOR_UI

        if state == States.WAIT_FOR_UI:
            self.get_logger().info(
                'Waiting for UI ball selection...'
            )

            self._shot_event.clear()
            self._reset_event.clear()

            while rclpy.ok():
                if self._shot_event.wait(timeout=0.1):
                    break

                if self._reset_event.is_set():
                    self.get_logger().info(
                        'Reset: going back to photo pose'
                    )
                    return States.MOVE_TO_PHOTO

            if not rclpy.ok():
                return States.FINISH

            with self._lock:
                if self.ui_shot_info is None:
                    self.get_logger().error(
                        'Shot event was set without target data'
                    )
                    return None

                shot_info = list(self.ui_shot_info)

            (
                target_x,
                target_y,
                target_z,
                white_x,
                white_y,
                white_z,
                force,
            ) = shot_info

            self.test_target_pose = {
                'target_x': target_x,
                'target_y': target_y,
                'target_z': target_z,
                'white_x': white_x,
                'white_y': white_y,
                'white_z': white_z,
                'force': force,
            }

            self.get_logger().info(
                'Position test target received: '
                f'target Base=({target_x:.2f}, '
                f'{target_y:.2f}, '
                f'{target_z:.2f}), '
                f'white Base=({white_x:.2f}, '
                f'{white_y:.2f}, '
                f'{white_z:.2f})'
            )

            return States.TEST_MOVE_ABOVE

        if state == States.TEST_MOVE_ABOVE:
            if self.test_target_pose is None:
                self.get_logger().error(
                    'Missing selected Base coordinate'
                )
                return None

            target_x = float(
                self.test_target_pose['target_x']
            )
            target_y = float(
                self.test_target_pose['target_y']
            )
            target_z = float(
                self.test_target_pose['target_z']
            )
            white_x = float(
                self.test_target_pose['white_x']
            )
            white_y = float(
                self.test_target_pose['white_y']
            )
            white_z = float(
                self.test_target_pose['white_z']
            )

            direction_x = target_x - white_x
            direction_y = target_y - white_y
            direction_length = math.hypot(
                direction_x,
                direction_y,
            )

            if direction_length < 1.0:
                self.get_logger().error(
                    'White ball and target ball are too close '
                    'to define a direction.'
                )
                return None

            unit_x = direction_x / direction_length
            unit_y = direction_y / direction_length
            base_yaw_deg = math.degrees(
                math.atan2(direction_y, direction_x)
            )
            target_rz = normalize_angle_deg(
                base_yaw_deg + TOOL_RZ_OFFSET_DEG
            )

            self.get_logger().info(
                '[BASE DIRECTION]\n'
                f'White Base=('
                f'{white_x:.2f}, {white_y:.2f}, {white_z:.2f})\n'
                f'Target Base=('
                f'{target_x:.2f}, {target_y:.2f}, {target_z:.2f})\n'
                f'Direction=({direction_x:.4f}, {direction_y:.4f})\n'
                f'Unit direction=({unit_x:.6f}, {unit_y:.6f})\n'
                f'Base yaw={base_yaw_deg:.2f}\n'
                f'Tool Rz offset={TOOL_RZ_OFFSET_DEG:.2f}\n'
                f'Command Rz={target_rz:.2f}'
            )

            response = self._call(
                self._req(
                    cmd_mode=RobotCommand.Request.CHECK_POSE,
                    tool=CUE_TOOL,
                )
            )

            if response is None:
                self.get_logger().error(
                    'Cannot read cue-tool pose'
                )
                return None

            current_tool_pose = list(response.current_position)

            # Rx/Ry must be confirmed using the teach pendant so the cue stays
            # in a safe, reasonable posture. They are not derived from the
            # white-to-target direction vector.
            target_rx = current_tool_pose[3]
            target_ry = current_tool_pose[4]

            transit_pose = self._twist(
                SAFE_TRANSIT_X,
                SAFE_TRANSIT_Y,
                SAFE_TRANSIT_Z,
                target_rx,
                target_ry,
                target_rz,
            )

            self.get_logger().info(
                '[SAFE TRANSIT]\n'
                f'Tool={CUE_TOOL} Base=0\n'
                f'X={SAFE_TRANSIT_X:.2f}\n'
                f'Y={SAFE_TRANSIT_Y:.2f}\n'
                f'Z={SAFE_TRANSIT_Z:.2f}\n'
                f'Rx={target_rx:.2f}\n'
                f'Ry={target_ry:.2f}\n'
                f'Rz={target_rz:.2f}\n'
                f'Velocity=10\n'
                f'Acceleration=10'
            )

            response = self._call(
                self._req(
                    cmd_mode=RobotCommand.Request.PTP,
                    tool=CUE_TOOL,
                    base=0,
                    pose=transit_pose,
                    velocity=10,
                    acceleration=10,
                )
            )

            if (
                response is None
                or response.arm_state != RobotCommand.Response.IDLE
            ):
                self.get_logger().error(
                    'Safe transit move failed'
                )
                return None

            transit_pose_response = self._call(
                self._req(
                    cmd_mode=RobotCommand.Request.CHECK_POSE,
                    tool=CUE_TOOL,
                )
            )

            if transit_pose_response is None:
                self.get_logger().error(
                    'Cannot read cue-tool pose after safe transit'
                )
                return None

            transit_tool_pose = list(
                transit_pose_response.current_position
            )

            command_x = target_x
            command_y = target_y
            command_z = target_z + TEST_ABOVE_OFFSET_MM

            target_pose = self._twist(
                command_x,
                command_y,
                command_z,
                transit_tool_pose[3],
                transit_tool_pose[4],
                transit_tool_pose[5],
            )

            self.get_logger().info(
                '[MOVE ABOVE BALL]\n'
                f'Tool={CUE_TOOL} Base=0\n'
                f'X={command_x:.2f}\n'
                f'Y={command_y:.2f}\n'
                f'Z={command_z:.2f}\n'
                f'Rx={transit_tool_pose[3]:.2f}\n'
                f'Ry={transit_tool_pose[4]:.2f}\n'
                f'Rz={transit_tool_pose[5]:.2f}\n'
                f'Ball Base Z={target_z:.2f}\n'
                f'Z offset={TEST_ABOVE_OFFSET_MM:.2f}\n'
                f'Velocity={TEST_VELOCITY}\n'
                f'Acceleration={TEST_ACCELERATION}'
            )

            response = self._call(
                self._req(
                    cmd_mode=RobotCommand.Request.PTP,
                    tool=CUE_TOOL,
                    base=0,
                    pose=target_pose,
                    velocity=TEST_VELOCITY,
                    acceleration=TEST_ACCELERATION,
                )
            )

            if response is None:
                self.get_logger().error(
                    'Position test movement failed'
                )
                return None

            result_pose = list(response.current_position)

            self.get_logger().info(
                '[FINAL TOOL POSE]\n'
                f'X={result_pose[0]:.2f}\n'
                f'Y={result_pose[1]:.2f}\n'
                f'Z={result_pose[2]:.2f}\n'
                f'Rx={result_pose[3]:.2f}\n'
                f'Ry={result_pose[4]:.2f}\n'
                f'Rz={result_pose[5]:.2f}\n'
                f'arm_state={response.arm_state}'
            )

            if response.arm_state != RobotCommand.Response.IDLE:
                self.get_logger().error(
                    'Robot did not reach test position, '
                    f'arm_state={response.arm_state}'
                )
                return None

            with self._lock:
                self.current_pose = result_pose

            self.get_logger().info(
                'Position test completed. '
                'Robot is stopped above the selected ball. '
                'Press Reset / Watch to return to the photo pose.'
            )

            self._reset_event.clear()

            while rclpy.ok():
                if self._reset_event.wait(timeout=0.1):
                    self.get_logger().info(
                        'Position test reset requested: '
                        'returning to photo pose'
                    )
                    return States.MOVE_TO_PHOTO

            return States.FINISH

        if state == States.FINISH:
            return States.FINISH

        self.get_logger().error(f'Unknown state: {state}')
        return None

    def _main_loop(self):
        state = States.INIT

        while (
            state not in (States.FINISH, None)
            and rclpy.ok()
        ):
            try:
                state = self._sm(state)
            except Exception as error:
                self.get_logger().error(
                    f'State machine error: {error}'
                )
                state = None

        self.get_logger().info('Arm loop ended')

    def start(self):
        Thread(
            target=self._main_loop,
            daemon=True,
            name='arm_loop',
        ).start()

    # ── Service helpers ───────────────────────────
    @staticmethod
    def _twist(
        linear_x,
        linear_y,
        linear_z,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    ):
        pose = Twist()
        pose.linear.x = float(linear_x)
        pose.linear.y = float(linear_y)
        pose.linear.z = float(linear_z)
        pose.angular.x = float(angular_x)
        pose.angular.y = float(angular_y)
        pose.angular.z = float(angular_z)
        return pose

    def _req(
        self,
        holding=True,
        cmd_mode=RobotCommand.Request.PTP,
        cmd_type=RobotCommand.Request.POSE_CMD,
        velocity=DEFAULT_VELOCITY,
        acceleration=DEFAULT_ACCEL,
        tool=1,
        base=0,
        digital_input_pin=0,
        digital_output_pin=0,
        digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
        pose=None,
        joints=None,
        circ_s=None,
        circ_end=None,
        jog_joint=6,
        jog_dir=0,
    ):
        if pose is None:
            pose = Twist()
        if joints is None:
            joints = [float('inf')] * 6
        if circ_s is None:
            circ_s = []
        if circ_end is None:
            circ_end = []

        request = RobotCommand.Request()
        request.digital_input_pin = digital_input_pin
        request.digital_output_pin = digital_output_pin
        request.digital_output_cmd = digital_output_cmd
        request.acceleration = acceleration
        request.jog_joint = jog_joint
        request.velocity = velocity
        request.tool = tool
        request.base = base
        request.cmd_mode = cmd_mode
        request.cmd_type = cmd_type
        request.circ_end = circ_end
        request.jog_dir = jog_dir
        request.holding = holding
        request.joints = joints
        request.circ_s = circ_s
        request.pose = pose
        return request

    def _call(self, request):
        while not self.hiwin_client.wait_for_service(
            timeout_sec=2.0
        ):
            if not rclpy.ok():
                return None

            self.get_logger().info(
                'Waiting for hiwinmodbus_service...'
            )

        future = self.hiwin_client.call_async(request)
        start_time = time.time()

        while not future.done():
            time.sleep(0.01)

            if not rclpy.ok():
                return None

            if time.time() - start_time > 30.0:
                self.get_logger().error(
                    'Service call timed out'
                )
                return None

        return future.result()


# ═══════════════════════════════════════════════════
#  Shot Planner UI
# ═══════════════════════════════════════════════════
class BilliardsUI:

    def __init__(self, root, node: Hiwin_Controller):
        self.root  = root
        self.node  = node
        root.title('Hiwin Billiards Shot Planner')
        root.configure(bg='#1a1a1a')
        root.resizable(False, False)

        # UI state
        self.ui_state      = UI_WATCH
        self.frozen_balls  = []
        self.white_ball    = None
        self.target_ball   = None
        self.target_pocket = None
        self.best_angle    = 0.0
        self.ghost_pos     = None
        self._drag_white   = False
        self._watching     = True
        self._manual_mode  = False   # False=auto-angle, True=manual drag angle

        self._build_ui()
        self._refresh()

    # ── Build UI ──────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg='#111', height=36)
        top.pack(fill='x')
        self.lbl_state = tk.Label(top, text='WATCHING', bg='#111',
            fg='#27ae60', font=('Helvetica', 11, 'bold'), padx=12)
        self.lbl_state.pack(side='left', pady=6)
        self.lbl_hint = tk.Label(top, text='Waiting for /ball_coordinate...',
            bg='#111', fg='#888', font=('Helvetica', 10), padx=8)
        self.lbl_hint.pack(side='left', pady=6)

        # Mode toggle (top right)
        self.mode_var = tk.BooleanVar(value=False)
        self.btn_mode = tk.Checkbutton(top,
            text='Manual angle', variable=self.mode_var,
            bg='#111', fg='#aaa', selectcolor='#333',
            activebackground='#111', font=('Helvetica', 10),
            command=self._on_mode_change)
        self.btn_mode.pack(side='right', padx=12, pady=6)

        main = tk.Frame(self.root, bg='#1a1a1a')
        main.pack(fill='both', expand=True, padx=8, pady=8)

        # Canvas
        self.canvas = tk.Canvas(main, width=UW, height=UH,
            bg='#155c2c', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, rowspan=30, padx=(0, 10))
        self.canvas.bind('<ButtonPress-1>',   self._on_click)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        # Right panel
        rp = tk.Frame(main, bg='#1a1a1a', width=210)
        rp.grid(row=0, column=1, sticky='nw')
        rp.grid_columnconfigure(0, weight=1)

        def sec(t, r):
            tk.Label(rp, text=t, bg='#1a1a1a', fg='#555',
                     font=('Helvetica', 9)).grid(
                         row=r, column=0, columnspan=2, sticky='w', pady=(10, 2))

        def sldr(r, var, lo, hi, val):
            s = tk.Scale(rp, variable=var, from_=lo, to=hi, orient='horizontal',
                         length=190, bg='#222', fg='#eee', troughcolor='#444',
                         highlightthickness=0,
                         command=lambda _: self._on_param_change())
            s.set(val)
            s.grid(row=r, column=0, columnspan=2, pady=2)
            return s

        sec('FORCE', 0)
        self.v_force = tk.IntVar(value=6)
        sldr(1, self.v_force, 1, 10, 6)

        sec('SPIN', 2)
        self.v_spin = tk.IntVar(value=0)
        sldr(3, self.v_spin, -5, 5, 0)

        sec('MANUAL ANGLE (deg)', 4)
        self.v_angle = tk.IntVar(value=0)
        self.sldr_angle = sldr(5, self.v_angle, 0, 359, 0)
        self.sldr_angle.config(state='disabled')  # enabled in manual mode

        self.v_bounce = tk.BooleanVar(value=True)
        tk.Checkbutton(rp, text='Show bounces', variable=self.v_bounce,
            bg='#1a1a1a', fg='#aaa', selectcolor='#333',
            activebackground='#1a1a1a',
            command=self._draw).grid(row=6, column=0, columnspan=2,
                                     sticky='w', pady=4)

        ttk.Separator(rp).grid(row=7, column=0, columnspan=2,
                               sticky='ew', pady=6)
        sec('TARGET POCKET', 8)
        self.pocket_var = tk.StringVar(value='')
        for i, (px, py, name, _) in enumerate(POCKET_POSITIONS):
            tk.Radiobutton(rp, text=name, variable=self.pocket_var, value=name,
                bg='#1a1a1a', fg='#aaa', selectcolor='#27ae60',
                activebackground='#1a1a1a',
                command=self._on_pocket_select).grid(
                    row=9 + i//3, column=i % 3, sticky='w', padx=4)

        ttk.Separator(rp).grid(row=11, column=0, columnspan=2,
                               sticky='ew', pady=6)
        sec('SHOT INFO', 12)
        self.lbl_angle  = self._info_row(rp, 'Angle',  13)
        self.lbl_force2 = self._info_row(rp, 'Force',  14)
        self.lbl_spin2  = self._info_row(rp, 'Spin',   15)
        self.lbl_pred   = self._info_row(rp, 'Pocket', 16)
        self.lbl_wb     = self._info_row(rp, 'W-ball', 17)

        ttk.Separator(rp).grid(row=18, column=0, columnspan=2,
                               sticky='ew', pady=6)

        def big_btn(text, color, row, cmd, state='normal'):
            b = tk.Button(rp, text=text, bg=color, fg='#fff',
                          font=('Helvetica', 10, 'bold'), relief='flat',
                          padx=6, pady=6, command=cmd, state=state)
            b.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
            return b

        self.btn_freeze = big_btn('Freeze Balls',      '#f39c12', 19, self._freeze_balls)
        self.btn_auto   = big_btn('Auto Best Shot',    '#2980b9', 20, self._auto_calculate)
        self.btn_send   = big_btn('MOVE ABOVE BALL ▶', '#27ae60', 21, self._send_shot,
                                  state='disabled')
        self.btn_reset  = big_btn('Reset / Watch',     '#555555', 22, self._reset)

        self.status = tk.Label(self.root, text='Waiting for /ball_coordinate...', anchor='w',
            bg='#111', fg='#888', font=('Courier', 9), padx=8)
        self.status.pack(fill='x', side='bottom')

        self.lbl_pose = tk.Label(self.root, text='Arm pos: --', anchor='w',
            bg='#0a0a0a', fg='#27ae60', font=('Courier', 9), padx=8)
        self.lbl_pose.pack(fill='x', side='bottom')

    def _info_row(self, parent, label, row):
        tk.Label(parent, text=label+':', bg='#1a1a1a', fg='#555',
                 font=('Helvetica', 9)).grid(row=row, column=0, sticky='w', padx=4)
        lbl = tk.Label(parent, text='--', bg='#1a1a1a', fg='#eee',
                       font=('Helvetica', 9, 'bold'))
        lbl.grid(row=row, column=1, sticky='w', padx=4)
        return lbl

    # ── Mode switch ───────────────────────────────
    def _on_mode_change(self):
        self._manual_mode = self.mode_var.get()
        if self._manual_mode:
            self.sldr_angle.config(state='normal')
            self._set_state('MANUAL MODE', 'Set angle with slider, click ball', '#e67e22')
        else:
            self.sldr_angle.config(state='disabled')
            self._set_state('AUTO MODE', 'Click target ball then pocket', '#2980b9')
        if self.ui_state == UI_PREVIEW:
            self._calculate_shot()
        self._draw()

    # ── Ball freeze ───────────────────────────────
    def _freeze_balls(self):
        live = self.node.get_balls_for_ui()
        if not live:
            messagebox.showwarning('No balls', 'No ball coordinates received yet.')
            return
        self.frozen_balls = live
        self._watching = False
        self.white_ball = None
        for b in self.frozen_balls:
            if b['label'].lower() in ('white', 'cue', 'w'):
                self.white_ball = dict(b); break
        if not self.white_ball:
            self.white_ball = {'x': UW//2-80, 'y': UH//2, 'label': 'W'}
        self.ui_state = UI_SELECT
        self._set_state('SELECT BALL', 'Click any ball to test its position', '#f39c12')
        self._draw()
        self.status['text'] = f'Frozen {len(live)} Base-coordinate balls'

    # ── Pocket selection ──────────────────────────
    def _on_pocket_select(self):
        name = self.pocket_var.get()
        for px, py, n, _ in POCKET_POSITIONS:
            if n == name:
                self.target_pocket = (px, py, n); break
        if self.target_ball:
            self._calculate_shot()
        self._draw()

    def _on_param_change(self):
        if self.ui_state == UI_PREVIEW:
            self._calculate_shot()
        self._draw()

    # ── Shot calculation ──────────────────────────
    def _calculate_shot(self):
        if not (self.white_ball and self.target_ball and self.target_pocket):
            return
        wx, wy = self.white_ball['x'], self.white_ball['y']
        tx, ty = self.target_ball['x'], self.target_ball['y']
        px, py, _ = self.target_pocket
        force  = self.v_force.get()
        spin   = self.v_spin.get()
        bounce = self.v_bounce.get()

        if self._manual_mode:
            angle = float(self.v_angle.get())
            rad   = math.radians(angle)
            # ghost: white ball travels along manual angle
            dist  = math.hypot(tx-wx, ty-wy)
            gx    = wx + math.cos(rad)*dist
            gy    = wy + math.sin(rad)*dist
        else:
            angle, gx, gy = ui_calc_hit_angle(wx, wy, tx, ty, px, py)
            self.v_angle.set(int(angle) % 360)

        self.best_angle = angle
        self.ghost_pos  = (gx, gy)

        rad   = math.radians(angle)
        speed = force * 3.5
        wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
        dx, dy   = gx-wx, gy-wy
        dist     = max(1, math.hypot(dx, dy))
        nx_, ny_ = dx/dist, dy/dist
        spin_eff = spin * 0.35
        (tvx, tvy), _ = ui_collision(wvx, wvy, gx, gy, tx, ty)
        tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
        tp = ui_simulate_path(tx, ty, tvx, tvy, force*50, bounce)
        last  = tp[-1]
        best_p = min(POCKET_POSITIONS,
                     key=lambda p: math.hypot(p[0]-last[0], p[1]-last[1]))
        best_d  = math.hypot(best_p[0]-last[0], best_p[1]-last[1])

        self.lbl_angle['text']  = f'{angle:.1f}°'
        self.lbl_force2['text'] = str(force)
        self.lbl_spin2['text']  = str(spin)
        self.lbl_pred['text']   = best_p[2] if best_d < 60 else 'Miss'
        self.lbl_wb['text']     = f'{wx:.0f},{wy:.0f}px'

        self.ui_state = UI_PREVIEW
        self._set_state('PREVIEW', 'Review then click MOVE ABOVE BALL', '#2980b9')
        self.btn_send['state'] = 'normal'

    def _auto_calculate(self):
        if not self.frozen_balls:
            messagebox.showinfo('Freeze first', 'Press Freeze Balls first.'); return
        if not self.target_pocket:
            messagebox.showinfo('Select pocket', 'Select a pocket first.'); return
        balls = [b for b in self.frozen_balls
                 if b['label'].lower() not in ('white', 'cue', 'w')]
        if not balls:
            messagebox.showinfo('No balls', 'No target balls detected.'); return
        px, py, _ = self.target_pocket
        wx = self.white_ball['x'] if self.white_ball else UW//2
        wy = self.white_ball['y'] if self.white_ball else UH//2
        force = self.v_force.get(); spin = self.v_spin.get()
        best, best_d = None, 9999
        for b in balls:
            angle, gx, gy = ui_calc_hit_angle(wx, wy, b['x'], b['y'], px, py)
            rad = math.radians(angle); speed = force*3.5
            wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
            dx, dy = gx-wx, gy-wy; dist = max(1, math.hypot(dx, dy))
            nx_, ny_ = dx/dist, dy/dist; spin_eff = spin*0.35
            (tvx, tvy), _ = ui_collision(wvx, wvy, gx, gy, b['x'], b['y'])
            tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
            tp = ui_simulate_path(b['x'], b['y'], tvx, tvy, force*50, True)
            d  = math.hypot(tp[-1][0]-px, tp[-1][1]-py)
            if d < best_d:
                best_d = d; best = b
        self.target_ball = dict(best)
        self._calculate_shot()
        self._draw()
        self.status['text'] = (f'Auto: Ball {best["label"]} '
                               f'angle={self.best_angle:.1f}°')

    # ── Send shot ─────────────────────────────────
    def _send_shot(self):
        """Move the cue tool to a safe height above the selected Base coordinate."""
        if self.target_ball is None:
            messagebox.showerror(
                'No ball selected',
                'Freeze balls, then click a ball first.'
            )
            return

        if (
            self.white_ball is None
            or any(
                key not in self.white_ball
                for key in ('base_x', 'base_y', 'base_z')
            )
        ):
            messagebox.showerror(
                'White ball missing',
                'No white ball Base coordinate is available.'
            )
            return

        selected_ball = self.target_ball
        target_x = float(selected_ball['base_x'])
        target_y = float(selected_ball['base_y'])
        target_z = float(selected_ball['base_z'])
        white_x = float(self.white_ball['base_x'])
        white_y = float(self.white_ball['base_y'])
        white_z = float(self.white_ball['base_z'])
        command_z = target_z + TEST_ABOVE_OFFSET_MM

        self.node.get_logger().info(
            '[UI selected]\n'
            f'Ball={selected_ball["label"]}\n'
            f'Received Base=('
            f'{target_x:.2f}, {target_y:.2f}, {target_z:.2f})\n'
            f'White Base=('
            f'{white_x:.2f}, {white_y:.2f}, {white_z:.2f})\n'
            f'Arm command=('
            f'{target_x:.2f}, {target_y:.2f}, {command_z:.2f})\n'
            f'Z offset={TEST_ABOVE_OFFSET_MM:.2f}\n'
            f'Tool={CUE_TOOL}'
        )

        self.node.set_shot_from_ui(
            target_x,
            target_y,
            target_z,
            white_x,
            white_y,
            white_z,
            self.v_force.get(),
        )

        self.ui_state = UI_HITTING
        self._set_state(
            'POSITION TEST...',
            'Robot moving above selected ball',
            '#e74c3c',
        )
        self.btn_send['state'] = 'disabled'
        self.status['text'] = (
            f'Ball {selected_ball["label"]}: '
            f'Base=({target_x:.1f}, {target_y:.1f}, {target_z:.1f}), '
            f'command Z={command_z:.1f}'
        )

        # Do not auto-reset. The robot remains above the selected ball
        # until Reset / Watch is pressed.

    def _reset(self):
        self.node.request_photo_pose()
        self.ui_state = UI_WATCH
        self.frozen_balls  = []
        self.white_ball    = None
        self.target_ball   = None
        self.target_pocket = None
        self.ghost_pos     = None
        self._watching     = True
        self.pocket_var.set('')
        self.btn_send['state'] = 'disabled'
        for lbl in (self.lbl_angle, self.lbl_force2, self.lbl_spin2,
                    self.lbl_pred,  self.lbl_wb):
            lbl['text'] = '--'
        self._set_state('MOVING TO PHOTO...', 'Arm returning to camera position', '#e67e22')
        self._draw()

    def _set_state(self, s, h, c):
        self.lbl_state['text'] = s
        self.lbl_state['fg']   = c
        self.lbl_hint['text']  = h

    # ── Canvas interaction ────────────────────────
    def _on_click(self, event):
        mx, my = event.x, event.y
        if self.ui_state == UI_WATCH:
            return

        # Position-test mode: allow selecting ANY frozen ball, including white.
        if self.ui_state in (UI_SELECT, UI_PREVIEW):
            nearest = None
            nearest_d = float('inf')
            for b in self.frozen_balls:
                d = math.hypot(mx - b['x'], my - b['y'])
                if d < nearest_d:
                    nearest_d = d
                    nearest = b

            if nearest is not None and nearest_d <= BR + 10:
                self.target_ball = dict(nearest)
                self.ui_state = UI_PREVIEW
                self.btn_send['state'] = 'normal'
                self._set_state(
                    'BALL SELECTED',
                    'Press MOVE ABOVE SELECTED BALL',
                    '#2980b9'
                )
                self.status['text'] = (
                    f'Selected ball {nearest["label"]}: '
                    f'canvas=({nearest["x"]:.1f}, {nearest["y"]:.1f})'
                )
                self._draw()
                return

    def _on_drag(self, event):
        if self._drag_white and self.white_ball:
            self.white_ball['x'] = max(UPAD+BR+2, min(UW-UPAD-BR-2, event.x))
            self.white_ball['y'] = max(UPAD+BR+2, min(UH-UPAD-BR-2, event.y))
            if self.target_ball and self.target_pocket:
                self._calculate_shot()
            self._draw()

    def _on_release(self, _):
        self._drag_white = False

    # ── Drawing ───────────────────────────────────
    def _draw(self):
        c = self.canvas
        c.delete('all')
        # Table
        c.create_rectangle(UPAD//2, UPAD//2, UW-UPAD//2, UH-UPAD//2,
                            fill='#8B4513', outline='')
        c.create_rectangle(UPAD, UPAD, UW-UPAD, UH-UPAD,
                            fill='#155c2c', outline='#0d3d1c', width=1.5)
        c.create_line(UW//2, UPAD, UW//2, UH-UPAD, fill='#1a7a3a', dash=(4, 10))
        c.create_oval(UW//2-35, UH//2-35, UW//2+35, UH//2+35,
                      fill='', outline='#1a7a3a')
        # Pockets
        for px, py, name, _ in POCKET_POSITIONS:
            sel = self.target_pocket and self.target_pocket[2] == name
            c.create_oval(px-PR, py-PR, px+PR, py+PR,
                          fill='#ffff00' if sel else '#000',
                          outline='#ffcc00' if sel else '#333',
                          width=2 if sel else 1)
            c.create_text(px, py, text=name, fill='#ffcc00' if sel else '#555',
                          font=('Helvetica', 6, 'bold'))
        # Preview lines
        if self.ui_state == UI_PREVIEW and self.white_ball and self.target_ball:
            self._draw_preview(c)
        # Balls
        balls = (self.frozen_balls if not self._watching
                 else self.node.get_balls_for_ui())
        for b in balls:
            bx, by   = b['x'], b['y']
            label    = b['label']
            color    = BALL_COLORS.get(label.lower(), '#888888')
            is_white = label.lower() in ('white', 'cue', 'w')
            is_tgt   = (self.target_ball and
                        abs(bx-self.target_ball['x']) < 3 and
                        abs(by-self.target_ball['y']) < 3)
            if is_tgt:
                c.create_oval(bx-BR-6, by-BR-6, bx+BR+6, by+BR+6,
                              fill='', outline='#ff4444', width=2, dash=(3, 3))
            c.create_oval(bx-BR, by-BR, bx+BR, by+BR, fill=color,
                          outline='#cccccc' if is_white else '#333333', width=1.5)
            c.create_text(bx, by, text=label[:2],
                          fill='#333' if is_white else '#fff',
                          font=('Helvetica', 7, 'bold'))
        # White ball overlay
        if self.white_ball and not self._watching:
            wx, wy = self.white_ball['x'], self.white_ball['y']
            c.create_oval(wx-BR-2, wy-BR-2, wx+BR+2, wy+BR+2,
                          fill='#ffffff', outline='#aaaaaa', width=1.5)
            c.create_text(wx, wy,      text='W', fill='#888', font=('Helvetica', 7, 'bold'))
            c.create_text(wx, wy+BR+9, text='drag',fill='#888888',font=('Helvetica', 7))
        # Mode indicator
        mode_txt = '🔧 MANUAL' if self._manual_mode else '🤖 AUTO'
        c.create_text(UW-UPAD-4, UH-UPAD-8, anchor='se',
                      text=mode_txt, fill='#27ae60', font=('Helvetica', 8))
        count = len(balls)
        c.create_text(UPAD+4, UH-UPAD-8, anchor='sw',
                      text=f'{count} balls', fill='#1d7a3a', font=('Helvetica', 8))
        if self._watching and count == 0:
            c.create_text(UW//2, UH//2, text='Waiting for /ball_coordinate...',
                          fill='#1d7a3a', font=('Helvetica', 12))

    def _draw_preview(self, c):
        wx, wy = self.white_ball['x'], self.white_ball['y']
        tx, ty = self.target_ball['x'], self.target_ball['y']
        px, py, _ = self.target_pocket
        force  = self.v_force.get()
        spin   = self.v_spin.get()
        bounce = self.v_bounce.get()
        rad    = math.radians(self.best_angle)
        speed  = force * 3.5
        wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
        gx, gy   = self.ghost_pos if self.ghost_pos else (tx, ty)
        # Ghost ball
        c.create_oval(gx-BR, gy-BR, gx+BR, gy+BR,
                      fill='', outline='#888888', dash=(4, 4))
        c.create_line(wx, wy, gx, gy, fill='#555555', dash=(4, 8))
        c.create_line(tx, ty, px, py, fill='#883333', dash=(3, 6))
        dx, dy = gx-wx, gy-wy
        dist   = max(1, math.hypot(dx, dy))
        nx_, ny_ = dx/dist, dy/dist
        spin_eff = spin*0.35
        steps1 = max(1, int(dist/speed))
        wp = ui_simulate_path(wx, wy, wvx, wvy, steps1, False)
        self._path(c, wp, '#ffffff', 2, (6, 4))
        (tvx, tvy), (rvx, rvy) = ui_collision(wvx, wvy, gx, gy, tx, ty)
        tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
        tp = ui_simulate_path(tx, ty, tvx, tvy, force*50, bounce)
        self._path(c, tp, '#ff5555', 2.5)
        if len(tp) > 1:
            self._arrow(c, tp[-2], tp[-1], '#ff3333')
        wp2 = ui_simulate_path(gx, gy, rvx, rvy, force*30, bounce)
        self._path(c, wp2, '#ffff88', 1, (3, 6))
        ax = wx + math.cos(rad)*35
        ay = wy + math.sin(rad)*35
        c.create_line(wx, wy, ax, ay, fill='#ffffff', width=2.5)
        self._arrow(c, (wx, wy), (ax, ay), '#ffffff')

    def _path(self, c, path, color, width=1.5, dash=None):
        if len(path) < 2: return
        flat = [v for pt in path for v in pt]
        opts = dict(fill=color, width=width, smooth=True)
        if dash: opts['dash'] = dash
        c.create_line(*flat, **opts)

    def _arrow(self, c, tail, tip, color):
        dx, dy = tip[0]-tail[0], tip[1]-tail[1]
        mag = math.hypot(dx, dy)
        if mag == 0: return
        nx, ny = dx/mag, dy/mag
        ax, ay = tip
        c.create_polygon(ax, ay,
                         ax-nx*10+ny*4, ay-ny*10-nx*4,
                         ax-nx*10-ny*4, ay-ny*10+nx*4,
                         fill=color, outline='')

    def _refresh(self):
        if self._watching:
            self._draw()
        p = self.node.get_current_pose()
        if any(v != 0 for v in p):
            self.lbl_pose['text'] = (
                f'Arm pos:  X={p[0]:8.2f}  Y={p[1]:8.2f}  Z={p[2]:8.2f}'
                f'  Rx={p[3]:7.2f}  Ry={p[4]:7.2f}  Rz={p[5]:7.2f}  mm/deg')
        self.root.after(150, self._refresh)


# ═══════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = Hiwin_Controller()

    # Start ROS spinning first so service responses and /ball_coordinate callbacks work
    # while the arm is moving to the photo pose.
    ros_thread = Thread(target=lambda: rclpy.spin(node), daemon=True,
                        name='ros_spin')
    ros_thread.start()
    node.start()

    # Do not create Tkinter until arm.yaml -> armpos has been reached.
    while rclpy.ok():
        if node.photo_pose_ready.wait(timeout=0.1):
            break
        if node.startup_failed.is_set():
            node.get_logger().error('Startup stopped because photo-pose movement failed')
            node.destroy_node()
            rclpy.shutdown()
            return

    if not rclpy.ok():
        return

    # Tkinter must run in the main thread. It opens only after the move completes.
    root = tk.Tk()
    app  = BilliardsUI(root, node)

    def _shutdown(*_):
        """Called on SIGINT / SIGTERM — tears down ROS then closes the window."""
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        try:
            root.destroy()
        except Exception:
            pass

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _poll_ros():
        """Periodic check: close UI if ROS has already shut down."""
        if not rclpy.ok():
            try:
                root.destroy()
            except Exception:
                pass
            return
        root.after(500, _poll_ros)

    root.after(500, _poll_ros)

    try:
        root.mainloop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        os.system('clear')


if __name__ == '__main__':
    main()
