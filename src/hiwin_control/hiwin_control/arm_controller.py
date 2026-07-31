#!/usr/bin/env python3
import time
import json
import math
from enum import Enum
from threading import Thread

import rclpy

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.node import Node
from hiwin_interfaces.srv import RobotCommand

import hiwin_control.nine_ball_strat as table2


CUE_TOOL = 5

DEFAULT_VELOCITY = 70
DEFAULT_ACCELERATION = 70

LIGHT_PIN = 6
HITSOFT_PIN = 4
HITMID_PIN = 5
HITHEAVY_PIN = 1
HEAVY_PIN = 2

END_TURN_RIGHT = [90.00, 0.00, 0.00, 0.00, -90.00, 0.00]

# 拍照姿態仍由 arm.yaml 提供，因為手臂仍要移到固定拍照位置。
# 球座標本身不再在本程式內進行任何相機或手眼校正。
import os
import yaml

current_dir = os.getcwd()
file_path_yaml = os.path.join(
    current_dir,
    'src/hiwin_control/hiwin_control/arm.yaml'
)

print('arm yaml:', file_path_yaml)

with open(file_path_yaml, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

FIX_ABS_CAM = data['armpos']


class States(Enum):
    INIT = 0
    FINISH = 1
    MOVE_TO_PHOTO_POSE = 2
    LOCK_INFO = 3
    OPEN_SEC_IO = 5
    STRATEGY = 6
    HITPOINT_PITCH = 7
    CHECK_POSE = 8
    HITPOINT_ANGLE = 9
    HITPOINT_TOP = 10
    HITBALL_POSE = 11
    HITBALL = 12
    AF_HITPOINT_TOP = 13
    CLOSE_ROBOT = 14


def yaw_angle(vectorx, vectory):
    """
    由擊球向量計算手臂 yaw 角度。
    """
    vector_length = math.sqrt(vectorx ** 2 + vectory ** 2)

    if vector_length == 0:
        raise ValueError('Aim vector length is zero.')

    cos_value = (-vectory) / vector_length
    cos_value = max(-1.0, min(1.0, cos_value))

    rad = math.acos(cos_value)
    theta = math.degrees(rad)

    if vectorx >= 0:
        return theta, rad

    return -theta, -rad


class HiwinController(Node):
    def __init__(self):
        super().__init__('hiwin_controller')

        self.hiwin_client = self.create_client(
            RobotCommand,
            'hiwinmodbus_service'
        )

        # 直接接收已轉換完成的 Base 座標 JSON：
        # [
        #   {"label":"1", "x":100.0, "y":200.0, "z":-120.0},
        #   {"label":"white", "x":300.0, "y":400.0, "z":-120.0}
        # ]
        self.ball_coordinate_subscriber = self.create_subscription(
            String,
            '/ball_coordinate',
            self.ball_coordinate_callback,
            10
        )

        # 最新一包已完成校正的 Base 球座標
        self.all_ball_coordinates = []

        # 本次擊球流程鎖定的球座標
        self.ball_coordinate_buffer = []

        self.strategy_info = []
        self.current_pose = None
        self.current_tool_pose = None

        self.hitpointx = None
        self.hitpointy = None
        self.obstacle = 0
        self.score = 0

        self.main_loop_thread = None

    # ========================================================
    # ROS callbacks
    # ========================================================

    def ball_coordinate_callback(self, msg):
        """接收 /ball_coordinate 發布的 Base 座標 JSON。"""
        try:
            parsed = json.loads(msg.data)

            if not isinstance(parsed, list):
                raise ValueError('ball_coordinate must be a JSON list.')

            validated = []

            for index, ball in enumerate(parsed):
                if not isinstance(ball, dict):
                    raise ValueError(
                        f'Ball index {index} is not a JSON object.'
                    )

                required = ('label', 'x', 'y', 'z')
                missing = [key for key in required if key not in ball]

                if missing:
                    raise ValueError(
                        f'Ball index {index} missing fields: {missing}'
                    )

                validated.append({
                    'label': str(ball['label']),
                    'x': float(ball['x']),
                    'y': float(ball['y']),
                    'z': float(ball['z']),
                })

            self.all_ball_coordinates = validated

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.get_logger().error(
                f'Failed to parse /ball_coordinate: {exc}'
            )

    # ========================================================
    # 狀態機
    # ========================================================

    def _state_machine(self, state):
        if state == States.INIT:
            self.get_logger().info('MOVING TO PREPARE POSE...')

            print('prepare joint:', END_TURN_RIGHT)

            req = self.generate_robot_request(
                cmd_type=RobotCommand.Request.JOINTS_CMD,
                joints=END_TURN_RIGHT
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to move to prepare pose.')
                return None

            self.get_logger().info('INIT / WAIT FOR BUTTON')

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.READ_DI,
                digital_input_pin=1
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to read DI pin 1.')
                return None

            last_state = res.digital_state

            while rclpy.ok():
                res = self.call_hiwin(req)

                if res is None:
                    self.get_logger().error('Failed to read DI pin 1.')
                    return None

                current_state = res.digital_state

                if current_state != last_state:
                    break

                time.sleep(0.02)

            return States.MOVE_TO_PHOTO_POSE

        elif state == States.MOVE_TO_PHOTO_POSE:
            self.ball_coordinate_buffer = []
            self.get_logger().info(
                'TURNING LIGHT ON / MOVING TO CAMERA POSE...'
            )

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd=RobotCommand.Request.DIGITAL_ON,
                digital_output_pin=LIGHT_PIN
            )

            self.call_hiwin(req)

            pose = Twist()

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = FIX_ABS_CAM[0:3]

            [
                pose.angular.x,
                pose.angular.y,
                pose.angular.z
            ] = FIX_ABS_CAM[3:6]

            print('camera pose:', FIX_ABS_CAM)

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                pose=pose
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to move to camera pose.')
                return None

            if res.arm_state == RobotCommand.Response.IDLE:
                return States.LOCK_INFO

            return None

        elif state == States.LOCK_INFO:
            # 等待 /ball_coordinate 在拍照姿態下更新
            time.sleep(1.0)

            self.get_logger().info(
                'LOCKING BASE BALL COORDINATES...'
            )

            self.ball_coordinate_buffer = [
                dict(ball)
                for ball in self.all_ball_coordinates
            ]

            if not self.ball_coordinate_buffer:
                self.get_logger().error(
                    'No data received from /ball_coordinate.'
                )
                return States.INIT

            labels = [
                ball['label']
                for ball in self.ball_coordinate_buffer
            ]

            if 'white' not in labels:
                self.get_logger().error(
                    'Cue ball was not found in /ball_coordinate. '
                    'Left/right photo search is disabled.'
                )
                return States.INIT

            self.get_logger().info(
                f'Locked {len(self.ball_coordinate_buffer)} '
                'Base ball coordinates.'
            )

            print(
                'locked ball coordinates:',
                self.ball_coordinate_buffer
            )

            # 座標已是 Base 座標，不做任何相機校正或二次校正。
            # 不開啟手動選球 UI，直接進入原本的自動策略流程。
            return States.OPEN_SEC_IO

        elif state == States.OPEN_SEC_IO:
            self.get_logger().info('OPENING SECOND IO')

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd=RobotCommand.Request.DIGITAL_ON,
                digital_output_pin=HEAVY_PIN
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to open second IO.')
                return None

            if res.arm_state == RobotCommand.Response.IDLE:
                return States.STRATEGY

            return None

        elif state == States.STRATEGY:
            # 關閉拍照燈
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin=LIGHT_PIN
            )
            self.call_hiwin(req)

            self.get_logger().info(
                'CALCULATING PATH WITH NINE_BALL_STRAT...'
            )

            # /ball_coordinate 已經是 Base 座標。
            # 根據 label 明確分離白球，不再假設白球一定排在最後。
            cue_ball = None
            object_balls = []

            for ball in self.ball_coordinate_buffer:
                if ball['label'] == 'white':
                    cue_ball = ball
                else:
                    object_balls.append(ball)

            if cue_ball is None:
                self.get_logger().error(
                    'Cue ball is missing before strategy calculation.'
                )
                return States.INIT

            if not object_balls:
                self.get_logger().error(
                    'No object balls are available for strategy calculation.'
                )
                return States.INIT

            object_ball_x = [
                float(ball['x'])
                for ball in object_balls
            ]
            object_ball_y = [
                float(ball['y'])
                for ball in object_balls
            ]

            cuex = float(cue_ball['x'])
            cuey = float(cue_ball['y'])

            print(
                'strategy object labels:',
                [ball['label'] for ball in object_balls]
            )
            print('strategy object x:', object_ball_x)
            print('strategy object y:', object_ball_y)
            print('strategy cue:', cuex, cuey)

            try:
                # 原始程式的決策介面：
                # [bestscore, bestvx, bestvy, countobs,
                #  hitpointx, hitpointy]
                self.strategy_info = table2.main(
                    object_ball_x,
                    object_ball_y,
                    cuex,
                    cuey
                )

            except Exception as exc:
                self.get_logger().error(
                    f'nine_ball_strat failed: {exc}'
                )
                return States.INIT

            if (
                self.strategy_info is None
                or len(self.strategy_info) < 6
            ):
                self.get_logger().error(
                    'nine_ball_strat returned invalid strategy data: '
                    f'{self.strategy_info}'
                )
                return States.INIT

            print('strategy info:', self.strategy_info)

            return States.HITPOINT_PITCH

        elif state == States.HITPOINT_PITCH:
            self.obstacle = self.strategy_info[3]

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool=CUE_TOOL
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to check cue tool pose.')
                return None

            self.current_tool_pose = res.current_position

            self.get_logger().info(
                'MOVING PITCH ANGLE IF NEEDED...'
            )

            pose = Twist()

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = [0.0, 408.0, 132.0]

            if self.obstacle == 1:
                pose.angular.x = self.current_tool_pose[3]
                pose.angular.y = 20.0
                pose.angular.z = self.current_tool_pose[5]
            else:
                [
                    pose.angular.x,
                    pose.angular.y,
                    pose.angular.z
                ] = self.current_tool_pose[3:6]

            print('pitch pose:', pose)

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                tool=CUE_TOOL,
                pose=pose
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to move pitch pose.')
                return None

            return States.CHECK_POSE

        elif state == States.CHECK_POSE:
            self.get_logger().info('CHECKING CURRENT POSE...')

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to check current pose.')
                return None

            self.current_pose = res.current_position

            return States.HITPOINT_ANGLE

        elif state == States.HITPOINT_ANGLE:
            self.get_logger().info('TURNING YAW ANGLE...')

            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]

            vx = self.strategy_info[1]
            vy = self.strategy_info[2]

            try:
                yaw, _ = yaw_angle(vx, vy)

            except ValueError as exc:
                self.get_logger().error(str(exc))
                return States.INIT

            pose = Twist()

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = self.current_pose[0:3]

            [
                pose.angular.x,
                pose.angular.y
            ] = self.current_pose[3:5]

            pose.angular.z = yaw - 90.0

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                tool=CUE_TOOL,
                pose=pose
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to turn yaw angle.')
                return None

            return States.HITPOINT_TOP

        elif state == States.HITPOINT_TOP:
            self.score = self.strategy_info[0]
            self.obstacle = self.strategy_info[3]

            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]

            self.get_logger().info('MOVING TO HITPOINT TOP...')

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool=CUE_TOOL
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to check hitpoint pose.')
                return None

            self.current_tool_pose = res.current_position

            pose = Twist()

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = [
                self.hitpointx,
                self.hitpointy,
                -70.0
            ]

            [
                pose.angular.x,
                pose.angular.y,
                pose.angular.z
            ] = self.current_tool_pose[3:6]

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                tool=CUE_TOOL,
                pose=pose
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to move to hitpoint top.')
                return None

            self.get_logger().info(
                'HITBALL_POSE AND HITBALL ARE DISABLED. '
                'SKIPPING LOWERING AND HIT IO.'
            )

            return States.AF_HITPOINT_TOP

        elif state == States.HITBALL_POSE:
            self.get_logger().info('GOING TO HIT BALL...')

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool=CUE_TOOL
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to check hitball pose.')
                return None

            self.current_pose = res.current_position

            pose = Twist()

            if self.obstacle == 0:
                hit_z = -135.0
            else:
                hit_z = -122.445

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = [
                self.hitpointx,
                self.hitpointy,
                hit_z
            ]

            [
                pose.angular.x,
                pose.angular.y,
                pose.angular.z
            ] = self.current_pose[3:6]

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                tool=CUE_TOOL,
                pose=pose
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error('Failed to move to hitball pose.')
                return None

            if res.arm_state == RobotCommand.Response.IDLE:
                return States.HITBALL

            return None

        elif state == States.HITBALL:
            if self.score <= 3500:
                hit_pin = HITHEAVY_PIN
            elif self.score <= 5000:
                hit_pin = HITMID_PIN
            else:
                hit_pin = HITSOFT_PIN

            self.get_logger().info('OPENING HIT PIN')

            print('hit pin IO:', hit_pin)

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd=RobotCommand.Request.DIGITAL_ON,
                digital_output_pin=hit_pin
            )

            self.call_hiwin(req)

            self.get_logger().info(
                'MOVING BACK TO HITPOINT TOP...'
            )

            pose = Twist()

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = [
                self.hitpointx,
                self.hitpointy,
                -70.0
            ]

            [
                pose.angular.x,
                pose.angular.y,
                pose.angular.z
            ] = self.current_pose[3:6]

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                tool=CUE_TOOL,
                pose=pose
            )

            self.call_hiwin(req)

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin=hit_pin
            )

            self.call_hiwin(req)

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin=HEAVY_PIN
            )

            self.call_hiwin(req)

            return States.AF_HITPOINT_TOP

        elif state == States.AF_HITPOINT_TOP:
            self.get_logger().info(
                'TURNING YAW ANGLE TO HOME...'
            )

            pose = Twist()

            [
                pose.linear.x,
                pose.linear.y,
                pose.linear.z
            ] = [
                self.hitpointx,
                self.hitpointy,
                -70.0
            ]

            [
                pose.angular.x,
                pose.angular.y,
                pose.angular.z
            ] = [-180.0, 0.0, 90.0]

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                holding=False,
                tool=CUE_TOOL,
                pose=pose
            )

            res = self.call_hiwin(req)

            if res is None:
                self.get_logger().error(
                    'Failed to return yaw angle home.'
                )
                return None

            return States.INIT

        elif state == States.CLOSE_ROBOT:
            self.get_logger().info('CLOSING ROBOT...')

            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CLOSE
            )

            self.call_hiwin(req)

            return States.FINISH

        else:
            self.get_logger().error(
                f'Unsupported state: {state}'
            )
            return None

    def _main_loop(self):
        state = States.INIT

        while rclpy.ok() and state != States.FINISH:
            state = self._state_machine(state)

            if state is None:
                self.get_logger().error(
                    'State machine stopped because next state is None.'
                )
                break

        self.destroy_node()

    # ========================================================
    # Hiwin service helpers
    # ========================================================

    def _wait_for_future_done(self, future, timeout=-1):
        time_start = time.time()

        while not future.done():
            time.sleep(0.01)

            if timeout > 0 and time.time() - time_start > timeout:
                self.get_logger().error(
                    'Wait for service timeout!'
                )
                return False

        return True

    def generate_robot_request(
        self,
        holding=True,
        cmd_mode=RobotCommand.Request.PTP,
        cmd_type=RobotCommand.Request.POSE_CMD,
        velocity=DEFAULT_VELOCITY,
        acceleration=DEFAULT_ACCELERATION,
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
        jog_dir=0
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

    def call_hiwin(self, req):
        while rclpy.ok() and not self.hiwin_client.wait_for_service(
            timeout_sec=2.0
        ):
            self.get_logger().info(
                'Hiwin service not available, waiting again...'
            )

        if not rclpy.ok():
            return None

        future = self.hiwin_client.call_async(req)

        if not self._wait_for_future_done(future):
            return None

        try:
            return future.result()

        except Exception as exc:
            self.get_logger().error(
                f'Hiwin service call failed: {exc}'
            )
            return None

    def start_main_loop_thread(self):
        self.main_loop_thread = Thread(
            target=self._main_loop,
            daemon=True
        )
        self.main_loop_thread.start()


def main(args=None):
    rclpy.init(args=args)

    controller = HiwinController()
    controller.start_main_loop_thread()

    try:
        rclpy.spin(controller)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
