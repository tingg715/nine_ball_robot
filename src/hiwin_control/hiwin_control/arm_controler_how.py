#!/usr/bin/env python3
import time
import yaml
import configparser
import os

import rclpy
from enum import Enum
from threading import Thread
from rclpy.node import Node
from rclpy.task import Future
from typing import NamedTuple
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from hiwin_interfaces.srv import RobotCommand

import cv2
import numpy as np
import math
import quaternion as qtn
import hiwin_control.transformations as transformations
import hiwin_control.nine_ball_strat as table2
import hiwin_control.pool_strat_v2 as pool
import matplotlib.pyplot as plt

CUE_TOOL = 8

# 桌面在 base frame 的 Z。由 arm.yaml 推得（2026-07-29 重標定後）：
#   armpos_z 423.503 + TOOL_TO_CAM[2] 68.170 - zoff 695.0 = -203.327
# 與 arm.yaml 的 pot0~3 z (-203.329) 相符，差 0.002mm。
#
# 本檔寫死的 z 都是「桌面 + 固定偏移」，但各點偏移是分別用 pendant 實測定出來的，
# 不是從舊值統一平移而來 —— 桌面再變時要逐點重測，不要整批加減。
# 目前實測值（相對桌面 -203.327）：
#   PITCH_POSE_XYZ    z   14.682 = 桌面 +218.0   中間仰角點，空曠可自由加高
#   HITPOINT_TOP/退回 z -124.318 = 桌面  +79.0
#   擊球 obstacle==0  z -175.000 = 桌面  +28.3
#   擊球 obstacle==1  z -162.433 = 桌面  +40.9
#
# HITPOINT_PITCH: 先在這個空曠的中間點把仰角轉好，再移到球上方，避免桿子掃到別的球。
PITCH_POSE_XYZ = [0., 375., 14.682]  # 桌面 +218.0，純淨空點，可自由加高
PITCH_ANGLE_Y = 20.0  # 路徑有障礙時的抬桿仰角（度）；無障礙則沿用當前 Ry

DEFAULT_VELOCITY = 70
DEFAULT_ACCELERATION = 70

LIGHT_PIN = 6
# 擊球電磁閥(單控 / 彈簧復歸)。原本 SOFT/MID/HEAVY 是 4/5/1 三條不同氣壓的
# 氣路，由 score 挑一條；現在硬體只剩這一條，力道由氣壓調節器決定(程式碰不到)。
HIT_PIN = 5
HEAVY_PIN = 2

# 擊球電磁閥通電時間(秒) = 閥開啟時間 = 氣缸伸出停留時間。
# 單控閥斷電才靠彈簧復歸，所以這個值就是「氣缸伸出多久才縮回」。
# 必須 >= 氣缸跑完行程的時間；設太短是在氣缸還在加速時切斷氣源，
# 結果是擊球不完全而且每一桿都不一樣(不可重複)。
# 這不是力道旋鈕 —— 力道請調氣壓調節器。
HIT_ON_SEC = 0.2

# [1.5683861280265246, -0.0021305364693475553, -3.056812207597806]
# FIX_ABS_CAM = [36.326, 376.998, 411.897, 180.0, 0.0, 90.0]
# FIX_ABS_RIGHT_CAM = [186.326, 376.998, 411.897, 180.0, 0.0, 90.0]
# FIX_ABS_LEFT_CAM = [-114.326, 376.998, 411.897, 180.0, 0.0, 90.0]
END_TURN_RIGHT = [90.00, 0.00, 0.00, 0.00, -90.00, 0.00] #手臂TURN右
END_TURN_HALL=[0.00, 44.832, 408.491, -179.999, -30.972, 89.998]


# Camera intrinsics + distortion, from camera_calibration.ini (Azure Kinect).
# Used by pixel_mm_convert() for the calibrated pinhole back-projection.
CAMERA_MATRIX = np.array([
    [921.1464532814899, 0.0, 957.208934796569],
    [0.0, 919.785424897187, 553.1754854696636],
    [0.0, 0.0, 1.0],
])
# OpenCV order [k1, k2, p1, p2, k3]; the ini's t1/t2 are p1/p2.
DIST_COEFFS = np.array([
    0.08790006867482206,     # k1
    -0.062180853042901774,   # k2
    0.0023044976254808424,   # t1 (p1)
    -0.0008184342596589589,  # t2 (p2)
    0.02495957912562247,     # k3
])

_CAM_FX = CAMERA_MATRIX[0, 0]
_CAM_FY = CAMERA_MATRIX[1, 1]
_CAM_CX = CAMERA_MATRIX[0, 2]
_CAM_CY = CAMERA_MATRIX[1, 2]

# Intrinsics of the UNDISTORTED /camera_calib image. camera.py undistorts the
# full frame with cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX, DIST_COEFFS,
# (1920, 1080), alpha=1.0), which produces a new camera matrix that differs
# from CAMERA_MATRIX (fx/fy change ~5%). YOLO detects on /camera_calib, so the
# pixels reaching pixel_mm_convert() are already undistorted and must be
# back-projected with THIS matrix, not the raw CAMERA_MATRIX.
# Recompute these four numbers if the camera resolution or alpha ever change.
UNDIST_FX = 967.19430947
UNDIST_FY = 940.82140851
UNDIST_CX = 955.28346322
UNDIST_CY = 554.72595329

# Read tool to camera vector
current_dir = os.getcwd()
config = configparser.ConfigParser()
file_path_ini = current_dir + '/src/hiwin_control/hiwin_control/eye_in_hand_calibration.ini'
config.read(file_path_ini)
tm = config['hand_eye_calibration']
TOOL_TO_CAM = [
    float(tm['y'])*1000,
    float(tm['x'])*1000,
    -float(tm['z'])*1000,
]

# Read table pot hole position and camera to table height
file_path_yaml = current_dir + '/src/hiwin_control/hiwin_control/arm.yaml'
print(file_path_yaml)
with open(file_path_yaml, 'r') as file:
    data = yaml.safe_load(file)

FIX_ABS_CAM = data['armpos']
FIX_ABS_RIGHT_CAM = [150.0, 332.87, 425.164, -179.999, -0.002, 89.644]  # provisional, replace after jogging to the real right-photo pose
FIX_ABS_LEFT_CAM = [-150.0, 332.87, 425.164, -179.999, -0.002, 89.644]  # provisional, replace after jogging to the real left-photo pose
CAM_TO_TABLE = data['zoff']
# CAM_TO_TABLE = 475

# 球心離桌面的高度 = 球半徑(球徑 38mm)。相機看到的是球心,不是桌面上的點。
BALL_R = 19.0
# 反投影「球」的像素要用「相機 → 球心平面」的距離。用 CAM_TO_TABLE(到桌面)
# 會把球往外推:誤差 = 離畫面中心距離 x (695/676 - 1) = 2.81%,桌角約 18mm。
# 注意:算桌面高度的 self.table_z 仍然要用 CAM_TO_TABLE,不要一起換掉。
CAM_TO_BALL = CAM_TO_TABLE - BALL_R

# When False, skip the DYNAMIC_CALI visual-servo refinement on the MAIN path
# (cue ball visible in the first photo) and shoot using the first-photo
# coordinates directly. STRATEGY then uses the first-photo target/cue positions
# instead of the servo-refined ones. Set back to True to restore the old
# behaviour. NOTE: the side-photo fallback (LOCK_CUE, cue not in first photo)
# still uses DYNAMIC_CALI, because there is no first-photo coordinate for the
# cue in that case.
USE_DYNAMIC_CALI = False

class States(Enum):
    INIT = 0
    FINISH = 1
    MOVE_TO_PHOTO_POSE = 2
    DYNAMIC_CALI = 3
    STRATEGY = 4
    HITPOINT_TOP = 5
    HITPOINT_ANGLE = 6
    HITBALL_POSE = 7
    HITBALL = 8
    AF_HITPOINT_ANGLE = 9
    AF_HITPOINT_TOP = 10
    CHECK_POSE = 11
    CLOSE_ROBOT = 12
    OPEN_SEC_IO = 13
    LOCK_INFO = 14
    MOVE_TO_LOWER = 15
    FIX_LEFT_PHOTO_POSE = 16
    FIX_RIGHT_PHOTO_POSE = 17
    LOCK_CUE = 18
    STEP_CALI = 19
    HITPOINT_PITCH = 20


def check_mid_pose(all_ball_pose):
    mid_error = []
    for i in range(0,len(all_ball_pose),2):
        dev_x = all_ball_pose[i] - 1920/2
        dev_y = all_ball_pose[i+1] - 1080/2
        temp_error = math.sqrt((dev_x)**2+(dev_y)**2)
        mid_error.append(temp_error)
    # print("mid error:", mid_error)
    min_error_index = mid_error.index(min(mid_error))
    # print("min index:", min_error_index)
    mid_x = all_ball_pose[2*min_error_index]
    mid_y = all_ball_pose[2*min_error_index+1]

    return mid_x, mid_y

def mid_point_error(mid_ball: list) -> float:
    dev_x = mid_ball[0] - 1920/2
    dev_y = mid_ball[1] - 1080/2
    mid_error = math.sqrt((dev_x)**2+(dev_y)**2)

    return mid_error

def pixel_mm_convert(cam_to_table_h, pixels):
    # The incoming pixel is already undistorted: YOLO detects on the
    # /camera_calib image, which camera.py has undistorted for the full frame.
    # So we do NOT run cv2.undistortPoints here (that would double-correct).
    # We simply back-project with the undistorted image's intrinsics (UNDIST_*),
    # not the raw CAMERA_MATRIX.
    u = float(pixels[0])
    v = float(pixels[1])

    dev_x = (u - UNDIST_CX) * cam_to_table_h / UNDIST_FX
    dev_y = (v - UNDIST_CY) * cam_to_table_h / UNDIST_FY
    cam_to_ball_pose = [dev_x, dev_y, 1.0]

    return cam_to_ball_pose

def yaw_angle(vectorx, vectory):
    vectorlength = math.sqrt((vectorx**2)+(vectory**2))
    rad = math.acos((-1*vectory)/(vectorlength*1))
    theta = rad*180/math.pi
    if vectorx >= 0:
        return theta, rad
    elif vectorx < 0:
        return -theta, -rad

def convert_arm_pose(ball_pose, arm_pose):
    base2tool = np.array(arm_pose)
    base2tool[:3] /= 1000

    # tool to camera quaternion, from eye_in_hand_calibration.ini [qx, qy, qz, qw] / [x, y, z] (metres)
    tool2cam_quaternion = [0.003109262802720864, -0.002553016790632168, 0.7071215509442967, 0.7070805659754924]
    tool2cam_trans = [0.11155349816142887, 0.0022825871424820617, -0.06817001513263532]

    tool2cam_rot = qtn.as_rotation_matrix(np.quaternion(tool2cam_quaternion[3],
                                                        tool2cam_quaternion[0],
                                                        tool2cam_quaternion[1],
                                                        tool2cam_quaternion[2]))
    tool2cam_trans = np.array([tool2cam_trans])

    # print("tool2cam_trans = {}".format(tool2cam_trans))

    tool2cam_mat = np.append(tool2cam_rot, tool2cam_trans.T, axis=1)
    tool2cam_mat = np.append(tool2cam_mat, np.array([[0., 0., 0., 1.]]), axis=0)

    quat = transformations.quaternion_from_euler(base2tool[3]*3.14/180,
                                                 base2tool[4]*3.14/180,
                                                 base2tool[5]*3.14/180,axes= "sxyz")

    base2tool_rot = qtn.as_rotation_matrix(np.quaternion(quat[3],
                                                         quat[0],
                                                         quat[1],
                                                         quat[2]))

    base2tool_trans = np.array([base2tool[:3]])

    base2tool_mat = np.append(base2tool_rot, base2tool_trans.T, axis=1)
    base2tool_mat = np.append(base2tool_mat, np.array([[0., 0., 0., 1.]]), axis=0)


    cam2ball_trans = np.array([[ball_pose[0]/1000,
                                ball_pose[1]/1000,
                                ball_pose[2]]])
    unit_matrix = np.identity(3)
    cam2ball_mat = np.append(unit_matrix, cam2ball_trans.T, axis=1)
    cam2ball_mat = np.append(cam2ball_mat, np.array([[0., 0., 0., 1.]]), axis=0)

    base2cam_mat = np.matmul(base2tool_mat, tool2cam_mat)
    base2ball_mat = np.matmul(base2cam_mat, cam2ball_mat)

    ax, ay, az = transformations.euler_from_matrix(base2ball_mat)
    base2ball_translation = transformations.translation_from_matrix(base2ball_mat)*1000.0

    calibrated_ball_pose = [base2ball_translation[0],
                            base2ball_translation[1],
                            base2ball_translation[2],
                            ax*180/3.14,
                            ay*180/3.14,
                            az*180/3.14]

    return calibrated_ball_pose

class Hiwin_Controller(Node):
    def __init__(self):
        super().__init__('hiwin_controller')
        self.hiwin_client = self.create_client(RobotCommand, 'hiwinmodbus_service')
        self.yolo_subscriber = self.create_subscription(Float64MultiArray, 'center_data_coords', self.yolo_callback, 10)
        self.yolo_label_subscriber = self.create_subscription(String, 'center_data_labels', self.label_callback, 10)
        # self.stategy_subscriber = self.create_subscription(Float64MultiArray, 'strategy_hitpoint',self.strategy_callback, 10)
        self.object_pose = None
        self.object_cnt = 0
        self.all_ball_pose = []
        self.strategy_info = []
        self.ball_pose_buffer = []
        self.target_cue = []
        self.all_label = []
        self.label_buffer = []
        # self.fix_z = 90.
        self.fix_z = 85.
        self.table_z = FIX_ABS_CAM[2] + TOOL_TO_CAM[2] - CAM_TO_TABLE

    # def strategy_callback(self, msg):
    #     _ = msg.data

    def label_callback(self, msg):
        self.all_label = eval(msg.data)

    def yolo_callback(self, msg):
        if not msg.data:
            self.data_flag = 0
        else:
            self.all_ball_pose = msg.data
            self.target_cue = [self.all_ball_pose[:2], self.all_ball_pose[-2:]]
            self.data_flag = 1


    def _state_machine(self, state: States) -> States:
        if state == States.INIT:
            self.get_logger().info('MOVING TO PREPARE POSE...')
            # joints=[float('inf')]*6,
            print("fix cam joint", END_TURN_RIGHT)
            req = self.generate_robot_request(
                cmd_type=RobotCommand.Request.JOINTS_CMD,
                joints = END_TURN_RIGHT,
                #不想讓手臂停止才做事(下面if/else不用，直接給下一步的狀態就好)
                #hold=Flase
                )
            res = self.call_hiwin(req)
            """
            need to place arm in initial position.
            1) first joint 90 degrees to the left or right
            2) fifth joint is on top of first joint
            """
            self.get_logger().info('INIT/WAIT FOR BUTTON')
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.READ_DI,
                digital_input_pin = 1
            )
            res = self.call_hiwin(req)
            last_state = res.digital_state
            while True:
                res = self.call_hiwin(req)
                current_state = res.digital_state
                if current_state != last_state:
                    break
                else:
                    continue
            nest_state = States.MOVE_TO_PHOTO_POSE

        elif state == States.MOVE_TO_PHOTO_POSE:
            # value need to be reset every time
            self.updated_target_cue = []
            self.index = 0

            self.get_logger().info('TUNING LIGHTS ON/MOVING TO CAMERA POSE...')
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_ON,
                digital_output_pin = LIGHT_PIN
            )
            self.call_hiwin(req)
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = FIX_ABS_CAM[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = FIX_ABS_CAM[3:6]
            print("fix cam pose", FIX_ABS_CAM)
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                pose = pose
                )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.LOCK_INFO
            else:
                nest_state = None

        elif state == States.FIX_RIGHT_PHOTO_POSE:
            self.fix_check_point = []
            self.get_logger().info('MOVING TO RIGHT PHOTO POSE...')
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_ON,
                digital_output_pin = LIGHT_PIN
            )
            self.call_hiwin(req)
            self.fix_check_point = FIX_ABS_RIGHT_CAM
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = self.fix_check_point[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = self.fix_check_point[3:6]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                pose = pose
                )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.LOCK_CUE
            else:
                nest_state = None

        elif state == States.FIX_LEFT_PHOTO_POSE:
            self.fix_check_point = []
            self.get_logger().info('MOVING TO RIGHT PHOTO POSE...')
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_ON,
                digital_output_pin = LIGHT_PIN
            )
            self.call_hiwin(req)
            self.fix_check_point = FIX_ABS_LEFT_CAM
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = self.fix_check_point[0:3]
            [pose.angular.x, pose.angular.y, pose.angular.z] = self.fix_check_point[3:6]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                pose = pose
                )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.LOCK_CUE
            else:
                nest_state = None

        elif state == States.LOCK_CUE:

            time.sleep(0.3)
            self.get_logger().info('LOCKING CUE BALL INFO...')
            self.ball_pose_buffer = self.all_ball_pose
            self.label_buffer = self.all_label
            if 'white' in self.label_buffer:
                print("white ball seen")
                temp_ball_pose_mm = pixel_mm_convert(CAM_TO_TABLE, self.ball_pose_buffer[-2:])
                temp_actual_pose = convert_arm_pose(temp_ball_pose_mm, self.fix_check_point)
                self.ball_pose[1] = temp_actual_pose
                self.fix_check_point = FIX_ABS_CAM
                nest_state = States.DYNAMIC_CALI

            else:
                print("MOVING TO FIX LEFT POSE...")
                nest_state = States.FIX_LEFT_PHOTO_POSE

        elif state == States.LOCK_INFO:
            time.sleep(1)
            self.ball_pose = []
            self.get_logger().info('LOCKING INFO FOR STRATEGY AND CALIBRATION...')
            self.ball_pose_buffer = self.all_ball_pose
            self.label_buffer = self.all_label
            self.target_cue = [self.ball_pose_buffer[:2], self.ball_pose_buffer[-2:]]
            print("target and cue:", self.target_cue)
            if 'white' in self.label_buffer:
                print("Cue ball seen")
                for target in self.target_cue:
                    print("target:", target)
                    temp_ball_pose_mm = pixel_mm_convert(CAM_TO_TABLE, target)
                    temp_actual_pose = convert_arm_pose(temp_ball_pose_mm, FIX_ABS_CAM)
                    self.ball_pose.append(temp_actual_pose[0:2])
                nest_state = States.DYNAMIC_CALI if USE_DYNAMIC_CALI else States.STRATEGY

            else:
                # Cue ball missing from the first photo. Do NOT fall back to the
                # side-photo path: STRATEGY converts self.ball_pose_buffer with
                # FIX_ABS_CAM, so side-photo pixels would displace every ball by
                # ~150mm. Go home and wait for the operator instead.
                self.get_logger().warning(
                    'CUE BALL NOT DETECTED - returning to prepare pose. '
                    'Clear the obstruction and press the button to retry.'
                )
                req = self.generate_robot_request(
                    cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                    digital_output_cmd = RobotCommand.Request.DIGITAL_OFF,
                    digital_output_pin = LIGHT_PIN
                )
                self.call_hiwin(req)
                nest_state = States.INIT

        elif state == States.STEP_CALI:
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin = LIGHT_PIN
            )
            self.call_hiwin(req)
            if self.index < len(self.target_cue):
                self.get_logger().info('MOVING TO CALIBRATION POSE...')
                self.get_logger().info('Camera moving to index_{} ball'.format(self.index))
                pose = Twist()
                [pose.linear.x, pose.linear.y, pose.linear.z] = [self.ball_pose[self.index][0] - TOOL_TO_CAM[0],
                                                                 self.ball_pose[self.index][1] - TOOL_TO_CAM[1],
                                                                 self.fix_z]
                # change
                [pose.angular.x, pose.angular.y, pose.angular.z] = FIX_ABS_CAM[3:6]
                print("Pose:", pose)

                req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                pose = pose,
                )
                res = self.call_hiwin(req)
                if res.arm_state == RobotCommand.Response.IDLE:
                    print("UPDATING BALL POSITION...")

                req = self.generate_robot_request(cmd_mode=RobotCommand.Request.CHECK_POSE)
                res = self.call_hiwin(req)
                cali_point = res.current_position

                while self.data_flag == 0:
                    if self.data_flag == 1:
                        break
                    else:
                        continue
                mid_x, mid_y = check_mid_pose(self.all_ball_pose)
                mid_rela_cam = pixel_mm_convert(self.fix_z - abs(TOOL_TO_CAM[2]) + abs(self.table_z), [mid_x, mid_y])
                self.updated_target_cue.append(cali_point[0] + TOOL_TO_CAM[0] + mid_rela_cam[0])
                self.updated_target_cue.append(cali_point[1] + TOOL_TO_CAM[1] - mid_rela_cam[1])
                self.index += 1

                if res.arm_state == RobotCommand.Response.IDLE and self.index < len(self.target_cue):
                    nest_state = States.STEP_CALI
                else:
                    nest_state = States.STRATEGY


        elif state == States.DYNAMIC_CALI:
            Kp = 0.25 # Proportion constant, P controller
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin = LIGHT_PIN
            )
            self.call_hiwin(req)
            if self.index < len(self.target_cue):
                self.get_logger().info('MOVING TO CALIBRATION POSE...')
                self.get_logger().info('Camera moving to index_{} ball'.format(self.index))
                pose = Twist()
                [pose.linear.x, pose.linear.y, pose.linear.z] = [self.ball_pose[self.index][0] - TOOL_TO_CAM[0],
                                                                 self.ball_pose[self.index][1] - TOOL_TO_CAM[1],
                                                                 self.fix_z]
                # change
                [pose.angular.x, pose.angular.y, pose.angular.z] = FIX_ABS_CAM[3:6]
                print("Pose:", pose)

                req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.PTP,
                pose = pose,
                )
                res = self.call_hiwin(req)
                if res.arm_state == RobotCommand.Response.IDLE:
                    print("START DYNAMIC CALIBRATION")

                req = self.generate_robot_request(cmd_mode=RobotCommand.Request.CHECK_POSE)
                res = self.call_hiwin(req)
                second_photo = res.current_position

                while True:
                    mid_x, mid_y = check_mid_pose(self.all_ball_pose)
                    mid_error = mid_point_error([mid_x, mid_y])
                    ball_relative_cam = pixel_mm_convert(self.fix_z - abs(TOOL_TO_CAM[2]) + abs(self.table_z), [mid_x, mid_y])

                    # mid_error = mid_point_error(self.target_cue[self.index])
                    # ball_relative_cam = pixel_mm_convert(self.fix_z - abs(TOOL_TO_CAM[2]) + abs(self.table_z), self.target_cue[self.index])

                    [pose.linear.x, pose.linear.y, pose.linear.z] = [second_photo[0] + Kp*ball_relative_cam[0],
                                                                    second_photo[1] - Kp*ball_relative_cam[1],
                                                                    self.fix_z]
                    [pose.angular.x, pose.angular.y, pose.angular.z] = FIX_ABS_CAM[3:6]

                    req = self.generate_robot_request(
                    cmd_mode=RobotCommand.Request.PTP,
                    # holding = False,
                    pose = pose
                    )
                    res = self.call_hiwin(req)

                    second_photo[0] += Kp*ball_relative_cam[0]
                    second_photo[1] -= Kp*ball_relative_cam[1]

                    # finish calibration condition
                    if abs(mid_error) <= 3:
                        # cv2.destroyAllWindows()
                        break

                # update ball position
                ball_relative_cam = pixel_mm_convert(self.fix_z - abs(TOOL_TO_CAM[2]) + abs(self.table_z), self.target_cue[self.index])
                req = self.generate_robot_request(cmd_mode=RobotCommand.Request.CHECK_POSE)
                res = self.call_hiwin(req)
                update_ball = res.current_position
                self.updated_target_cue.append(update_ball[0] + TOOL_TO_CAM[0] + ball_relative_cam[0])
                self.updated_target_cue.append(update_ball[1] + TOOL_TO_CAM[1] - ball_relative_cam[1])

                self.index += 1

                if res.arm_state == RobotCommand.Response.IDLE and self.index < len(self.target_cue):
                    nest_state = States.DYNAMIC_CALI
                else:
                    nest_state = States.STRATEGY

        elif state == States.OPEN_SEC_IO:
            self.get_logger().info('Opening second IO\n')
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_ON,

                digital_output_pin = HEAVY_PIN
            )
            res = self.call_hiwin(req)
            # time.sleep(0.5)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.STRATEGY
            else:
                nest_state = None

        elif state == States.STRATEGY:
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin = LIGHT_PIN
            )
            self.call_hiwin(req)
            actual_x = []
            actual_y = []
            self.get_logger().info('UPDATE BALL POSITION')
            for i in range(0, len(self.ball_pose_buffer), 2):
                i_rela_to_cam = pixel_mm_convert(CAM_TO_BALL, self.ball_pose_buffer[i:i+2])
                actual_ball_pose = convert_arm_pose(i_rela_to_cam, FIX_ABS_CAM)
                actual_x.append(actual_ball_pose[0])
                actual_y.append(actual_ball_pose[1])
            print("before cali x:", actual_x)
            print("before cali y:", actual_y)
            print("\n")


            if self.updated_target_cue:
                # DYNAMIC_CALI ran: overwrite target ball (index 0) and cue ball
                # (index -1) with the servo-refined positions.
                actual_x[0] = self.updated_target_cue[0]
                actual_y[0] = self.updated_target_cue[1]
                actual_x[-1] = self.updated_target_cue[-2]
                actual_y[-1] =  self.updated_target_cue[-1]
                cuex, cuey = self.updated_target_cue[-2], self.updated_target_cue[-1]
            else:
                # DYNAMIC_CALI skipped (USE_DYNAMIC_CALI=False): keep the
                # first-photo coordinates already computed above for every ball,
                # including the cue (last entry).
                cuex, cuey = actual_x[-1], actual_y[-1]
            print("after cali x:", actual_x)
            print("after cali y:", actual_y)
            '''
            check before and after update
            '''
            ballcount = len(actual_x) - 1

            self.get_logger().info('CALCULATE PATH')
            # table.main returns -> [bestscore, bestvx, bestvy, countobs, final_self.hitpointx, final_self.hitpointy]
            self.strategy_info = table2.main(actual_x[:-1], actual_y[:-1], cuex, cuey)
            print("strategy info:", self.strategy_info)
            nest_state = States.HITPOINT_PITCH

        elif state == States.HITPOINT_PITCH:
            self.obstacle = self.strategy_info[3]
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool = CUE_TOOL,
                )
            res = self.call_hiwin(req)
            self.current_tool_pose = res.current_position
            self.get_logger().info('MOVING PITCH ANGLE IF ANY...')
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = PITCH_POSE_XYZ  # 寫死的中間仰角點
            pose.angular.x = self.current_tool_pose[3]
            pose.angular.y = PITCH_ANGLE_Y if self.obstacle == 1 else self.current_tool_pose[4]
            pose.angular.z = self.current_tool_pose[5]
            print("POSE:", pose)
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                # holding=False,
                tool = CUE_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)
            nest_state = States.CHECK_POSE

        elif state == States.HITPOINT_TOP:
            # 路徑分數。目前不參與任何決策(擊球固定用 HIT_PIN，力道看氣壓調節器)，
            # 留著是為了 debug 路徑選擇，也是將來要做力道分段時現成的輸入。
            self.score = self.strategy_info[0]
            self.obstacle = self.strategy_info[3]
            self.get_logger().info('MOVING TO HITPOINT TOP...')
            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]
            vx = self.strategy_info[1]
            vy = self.strategy_info[2]

            self.obstacle = self.strategy_info[3]
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool = CUE_TOOL,
                )
            res = self.call_hiwin(req)
            self.current_tool_pose = res.current_position
            # yaw, _ = yaw_angle(vx, vy)
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = [self.hitpointx, self.hitpointy, -124.318]  # 桌面 +79.0
            [pose.angular.x, pose.angular.y, pose.angular.z] = self.current_tool_pose[3:6]

            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                # holding=False,
                tool = CUE_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)
            nest_state = States.HITBALL_POSE
            # if res.arm_state == RobotCommand.Response.IDLE:
            #     nest_state = States.HITPOINT_ANGLE
            # else:
            #     nest_state = None

        elif state == States.HITPOINT_ANGLE:
            self.get_logger().info('TURNINING YAW ANGLE...')
            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]
            vx = self.strategy_info[1]
            vy = self.strategy_info[2]
            yaw, _ = yaw_angle(vx, vy)
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = self.current_pose[0:3]
            [pose.angular.x, pose.angular.y] = self.current_pose[3:5]
            pose.angular.z = yaw-90.
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                tool = CUE_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)
            nest_state = States.HITPOINT_TOP
            # if res.arm_state == RobotCommand.Response.IDLE:
            #     nest_state = States.HITPOINT_TOP
            # else:
            #     nest_state = None

        elif state == States.CHECK_POSE:
            self.get_logger().info('CHECK_POSE')
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                )
            res = self.call_hiwin(req)
            self.current_pose = res.current_position
            nest_state = States.HITPOINT_ANGLE
            # if res.arm_state == RobotCommand.Response.IDLE:
            #     nest_state = States.HITBALL_POSE
            # else:
            #     nest_state = None

        elif state == States.HITBALL_POSE:
            self.get_logger().info('GOING TO HIT BALL...')
            pose = Twist()
            req = self.generate_robot_request(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool = CUE_TOOL,
                # holding=False,
                )
            res = self.call_hiwin(req)
            self.current_pose = res.current_position
            if self.obstacle == 0:
                # ............................................
                [pose.linear.x, pose.linear.y, pose.linear.z] = [self.hitpointx, self.hitpointy, -175.0]
                # ............................................
            else:
                [pose.linear.x, pose.linear.y, pose.linear.z] = [self.hitpointx, self.hitpointy, -162.433]
            [pose.angular.x, pose.angular.y, pose.angular.z] = self.current_pose[3:6]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                tool = CUE_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)
            if res.arm_state == RobotCommand.Response.IDLE:
                nest_state = States.HITBALL
            else:
                nest_state = None

        elif state == States.HITBALL:
            self.get_logger().info('OPEN PIN TO HIT BALL')
            print("hit pin IO:", HIT_PIN, " on(s):", HIT_ON_SEC)
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_ON,
                digital_output_pin = HIT_PIN
            )
            self.call_hiwin(req)

            # 通電時間由 HIT_ON_SEC 決定，不是由退回動作的長度決定。
            # 之前 OFF 排在下面那個 PTP 之後，而 PTP 是 holding=True(阻塞)，
            # 單控閥會一路通電到手臂退回完成才斷電 —— 氣缸整段退回過程都是
            # 伸出狀態，桿子被手臂拖著走。
            time.sleep(HIT_ON_SEC)

            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin = HIT_PIN
            )
            res = self.call_hiwin(req)

            self.get_logger().info('MOVING BACK TO HIT POINT TOP WITH YAW ANGLE...')
            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]
            vx = self.strategy_info[1]
            vy = self.strategy_info[2]
            yaw, _ = yaw_angle(-vx, -vy)
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = [self.hitpointx, self.hitpointy, -124.318]
            [pose.angular.x, pose.angular.y, pose.angular.z] = self.current_pose[3:6]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                # holding 維持 default True: 退回動作要做完才進 INIT，
                # 否則會跟 INIT 的 END_TURN_RIGHT JOINTS_CMD 疊在一起。
                tool = CUE_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)

            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.DIGITAL_OUTPUT,
                digital_output_cmd = RobotCommand.Request.DIGITAL_OFF,
                digital_output_pin = HEAVY_PIN
            )
            res = self.call_hiwin(req)
            # Skip AF_HITPOINT_TOP (the extra yaw-to-home step); go straight back
            # to INIT, whose joint reset (END_TURN_RIGHT) re-homes the arm anyway.
            nest_state = States.INIT


        elif state == States.AF_HITPOINT_TOP:
            self.get_logger().info('TURNING YAW ANGLE TO HOME...')
            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]
            vx = self.strategy_info[1]
            vy = self.strategy_info[2]
            yaw, _ = yaw_angle(-vx, -vy)
            pose = Twist()
            [pose.linear.x, pose.linear.y, pose.linear.z] = [self.hitpointx, self.hitpointy, -124.318]
            [pose.angular.x, pose.angular.y, pose.angular.z] = [-180., 0., 90.]
            req = self.generate_robot_request(
                cmd_mode = RobotCommand.Request.PTP,
                holding=False,
                tool = CUE_TOOL,
                pose = pose
            )
            res = self.call_hiwin(req)
            nest_state = States.INIT
            # if res.arm_state == RobotCommand.Response.IDLE:
            #     nest_state = States.MOVE_TO_PHOTO_POSE
            # else:
            #     nest_state = None


        elif state == States.CLOSE_ROBOT:
            self.get_logger().info('CLOSE_ROBOT')
            req = self.generate_robot_request(cmd_mode=RobotCommand.Request.CLOSE)
            res = self.call_hiwin(req)
            nest_state = States.FINISH

        else:
            nest_state = None
            self.get_logger().error('Input state not supported!')
            # return
        return nest_state

    def _main_loop(self):
        state = States.INIT
        while state != States.FINISH:
            state = self._state_machine(state)
            if state == None:
                break
        self.destroy_node()

    def _wait_for_future_done(self, future: Future, timeout=-1):
        time_start = time.time()
        while not future.done():
            time.sleep(0.01)
            if timeout > 0 and time.time() - time_start > timeout:
                self.get_logger().error('Wait for service timeout!')
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
            pose=Twist(),
            joints=[float('inf')]*6,
            circ_s=[],
            circ_end=[],
            jog_joint=6,
            jog_dir=0
            ):
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
        while not self.hiwin_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('service not available, waiting again...')
        future = self.hiwin_client.call_async(req)
        if self._wait_for_future_done(future):
            res = future.result()
        else:
            res = None
        return res

    def start_main_loop_thread(self):
        self.main_loop_thread = Thread(target=self._main_loop)
        self.main_loop_thread.daemon = True
        self.main_loop_thread.start()

def main(args=None):
    rclpy.init(args=args)

    stratery = Hiwin_Controller()
    stratery.start_main_loop_thread()

    rclpy.spin(stratery)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
