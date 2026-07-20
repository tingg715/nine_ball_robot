#!/usr/bin/env python3

import os
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv2 import aruco
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.task import Future
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image

from hiwin_interfaces.srv import RobotCommand

import configparser

# ============================== CONFIG ==============================
# 桌面座標系：自行定義原點/方向，例如左下角=(0,0)，往右+X，往對面+Y。
# 要跟你實際貼 4 個標記的位置、ID 對應好。
TABLE_W = 1134.0   # TODO: 實際量測的桌寬 (mm)，比賽平面(庫邊內側,貼平桌面高度)1155-38
TABLE_H = 572.0    # TODO: 實際量測的桌長 (mm)

TAG_TABLE_POS = {   # 4 個標記各自在「桌面座標系」的位置 (mm)，僅供參考/未來 homography 用
    0: (0.0,     0.0),
    1: (TABLE_W, 0.0),
    2: (TABLE_W, TABLE_H),
    3: (0.0,     TABLE_H),
}

# 手臂能同時看到全部4個標記的固定拍照姿態。
# 手動 jog 手臂到一個能看到全桌4個標記的位置，pendant螢幕上顯示的座標直接
# 填在這裡（法蘭/TCP姿態，不用換算），程式會原封不動地用這組數字下PTP指令
# 移動過去，之後 arm_controller_v2.py 打球時也是直接用同一組數字移動回來。
PHOTO_ARM_POSE = [0.0, 332.870, 425.166, 180.0, -0.003, 90.0]  # TODO: jog後從pendant讀值填入 [x,y,z,rx,ry,rz]

# 相機光心到桌面平面的最短垂直距離（公尺）。
CAMERA_TO_TABLE_NORMAL_DISTANCE_M = 0.614

CAMERA_CALIBRATION_FILE = (
    '/home/ting/work/src/hiwin_control/'
    'hiwin_control/camera_calibration.ini'
)
HAND_EYE_CALIBRATION_FILE = (
    '/home/ting/work/src/hiwin_control/'
    'hiwin_control/eye_in_hand_calibration.ini'
)
DEBUG_IMAGE_FILE = '/tmp/aruco_table_cali_debug.jpg'

DEFAULT_VELOCITY = 50
DEFAULT_ACCELERATION = 10
# ======================================================================


def make_transform(rotation_matrix, translation):
    """建立 A_T_B；它會把 B frame 的點轉換到 A frame。"""
    rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if rotation_matrix.shape != (3, 3):
        raise ValueError('Rotation matrix must have shape (3, 3).')
    if translation.shape != (3,):
        raise ValueError('Translation must have shape (3,).')
    if not np.all(np.isfinite(rotation_matrix)):
        raise ValueError('Rotation matrix contains non-finite values.')
    if not np.all(np.isfinite(translation)):
        raise ValueError('Translation contains non-finite values.')

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation
    return transform


class TableCaliV3(Node):
    def __init__(self):
        super().__init__('table_cali_v3')
        self.hiwin_client = self.create_client(RobotCommand, 'hiwinmodbus_service')

        self.camera_matrix, self.dist_coeffs = self.read_camera_config(
            CAMERA_CALIBRATION_FILE
        )
        (
            self.tool_to_camera_translation,
            self.tool_to_camera_quaternion,
        ) = self.read_hand_eye_config(HAND_EYE_CALIBRATION_FILE)

        tool_to_camera_rotation = Rotation.from_quat(
            self.tool_to_camera_quaternion
        ).as_matrix()
        self.tool_to_camera = make_transform(
            tool_to_camera_rotation,
            self.tool_to_camera_translation,
        )

        self.get_logger().info(
            f'Camera calibration file path: {CAMERA_CALIBRATION_FILE}'
        )
        self.get_logger().info(f'Camera matrix:\n{self.camera_matrix}')
        self.get_logger().info(
            f'Distortion coefficients: {self.dist_coeffs.tolist()}'
        )
        self.get_logger().info(
            f'Hand-eye calibration file path: {HAND_EYE_CALIBRATION_FILE}'
        )
        self.get_logger().info(
            'Tool_T_Camera translation [m]: '
            f'{self.tool_to_camera_translation.tolist()}'
        )
        self.get_logger().info(
            'Tool_T_Camera quaternion [qx, qy, qz, qw]: '
            f'{self.tool_to_camera_quaternion.tolist()}'
        )
        self.get_logger().info(
            f'Tool_T_Camera matrix:\n{self.tool_to_camera}'
        )

        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, '/rgb/image_raw', self._img_cb, 1)

        self.aruco_dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.aruco_parameters = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        self.config = {}

    # ---------------- camera ----------------
    def _img_cb(self, msg):
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def capture_img(self):
        # 靠 main() 那邊的 rclpy.spin(node) 持續餵新的影格進 _img_cb，
        # 這裡（跑在背景執行緒）只需要輪詢等最新一張。
        while self.latest_frame is None:
            time.sleep(0.01)
        return self.latest_frame.copy()

    def read_camera_config(self, filepath):
        config = configparser.ConfigParser()

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f'Camera calibration file not found: {filepath}'
            )

        loaded_files = config.read(filepath)

        if not loaded_files:
            raise RuntimeError(
                f'Failed to read camera calibration file: {filepath}'
            )

        self.get_logger().info(
            f'Camera calibration file: {filepath}'
        )
        self.get_logger().info(
            f'Config sections: {config.sections()}'
        )

        if 'Intrinsic' not in config:
            raise KeyError(
                f'No [Intrinsic] section in {filepath}'
            )

        if 'Distortion' not in config:
            raise KeyError(
                f'No [Distortion] section in {filepath}'
            )

        cm = config['Intrinsic']
        dc = config['Distortion']

        camera_matrix = np.array([
            [
                float(cm['0_0']),
                float(cm['0_1']),
                float(cm['0_2'])
            ],
            [
                float(cm['1_0']),
                float(cm['1_1']),
                float(cm['1_2'])
            ],
            [
                float(cm['2_0']),
                float(cm['2_1']),
                float(cm['2_2'])
            ]
        ], dtype=np.float64)

        dist_coeffs = np.array([
            float(dc['k1']),
            float(dc['k2']),
            float(dc['t1']),
            float(dc['t2']),
            float(dc['k3'])
        ], dtype=np.float64)

        return camera_matrix, dist_coeffs

    def read_hand_eye_config(self, filepath):
        config = configparser.ConfigParser()
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f'Hand-eye calibration file not found: {filepath}'
            )

        loaded_files = config.read(filepath)
        if not loaded_files:
            raise RuntimeError(
                f'Failed to read hand-eye calibration file: {filepath}'
            )
        if 'hand_eye_calibration' not in config:
            raise KeyError(
                f'No [hand_eye_calibration] section in {filepath}'
            )

        section = config['hand_eye_calibration']
        required = ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')
        missing = [field for field in required if field not in section]
        if missing:
            raise KeyError(
                'Missing hand-eye calibration field(s): '
                + ', '.join(missing)
            )

        try:
            translation = np.array(
                [float(section[field]) for field in ('x', 'y', 'z')],
                dtype=np.float64,
            )
            quaternion = np.array(
                [float(section[field]) for field in ('qx', 'qy', 'qz', 'qw')],
                dtype=np.float64,
            )
        except ValueError as exc:
            raise ValueError(
                'Hand-eye x, y, z, qx, qy, qz, qw must be floats.'
            ) from exc

        if not np.all(np.isfinite(translation)):
            raise ValueError('Hand-eye translation contains non-finite values.')
        if not np.all(np.isfinite(quaternion)):
            raise ValueError('Hand-eye quaternion contains non-finite values.')

        quaternion_norm = float(np.linalg.norm(quaternion))
        if quaternion_norm < 1e-12:
            raise ValueError(
                'Hand-eye quaternion norm is too close to zero.'
            )
        quaternion /= quaternion_norm
        return translation, quaternion

    def undistort_pixel(self, pixel):
        pixel = np.asarray(pixel, dtype=np.float64)
        if pixel.shape != (2,) or not np.all(np.isfinite(pixel)):
            raise ValueError('Pixel must contain two finite values [u, v].')

        points = pixel.reshape(1, 1, 2)
        corrected = cv2.undistortPoints(
            points,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix,
        )
        return corrected[0, 0].astype(np.float64)

    # ---------------- arm ----------------
    def generate_robot_request(self, holding=True, cmd_mode=RobotCommand.Request.PTP,
                                cmd_type=RobotCommand.Request.POSE_CMD,
                                velocity=DEFAULT_VELOCITY, acceleration=DEFAULT_ACCELERATION,
                                tool=0, base=0, digital_input_pin=0, digital_output_pin=0,
                                digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
                                pose=Twist(), joints=[float('inf')] * 6,
                                circ_s=[], circ_end=[], jog_joint=6, jog_dir=0):
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

    def _wait_for_future_done(self, future: Future, timeout=-1):
        time_start = time.time()
        while not future.done():
            time.sleep(0.01)
            if timeout > 0 and time.time() - time_start > timeout:
                return False
        return True

    def call_hiwin(self, req):
        while not self.hiwin_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('waiting for hiwinmodbus_service...')
        future = self.hiwin_client.call_async(req)
        if self._wait_for_future_done(future):
            return future.result()
        return None

    # ---------------- aruco helpers ----------------
    def detect(self, img):
        corners, ids, _ = self.aruco_detector.detectMarkers(img)
        ids = [] if ids is None else ids.flatten()
        corners = [c.reshape((4, 2)) for c in corners]
        idmap = {tag_id: i for i, tag_id in enumerate(ids)}
        return idmap, corners

    def selected_corner(self, tag_id, corner):
        (tl, tr, br, bl) = corner
        corner_map = {
            0: bl,
            1: br,
            2: tr,
            3: tl,
        }
        if tag_id not in corner_map:
            raise ValueError(
                f'No selected ArUco corner configured for ID {tag_id}.'
            )
        selected = corner_map[tag_id]
        return (
            float(selected[0]),
            float(selected[1]),
        )

    def save_detection_debug_image(self, image, idmap, corners):
        debug_image = image.copy()
        if idmap:
            detected_ids = list(idmap.keys())
            draw_corners = [
                corners[idmap[tag_id]].reshape(1, 4, 2).astype(np.float32)
                for tag_id in detected_ids
            ]
            draw_ids = np.asarray(
                detected_ids,
                dtype=np.int32,
            ).reshape(-1, 1)
            aruco.drawDetectedMarkers(debug_image, draw_corners, draw_ids)

        selected_names = {
            0: 'bl',
            1: 'br',
            2: 'tr',
            3: 'tl',
        }
        for tag_id in TAG_TABLE_POS:
            if tag_id not in idmap:
                continue
            selected = self.selected_corner(
                tag_id,
                corners[idmap[tag_id]],
            )
            selected_pixel = (
                int(round(selected[0])),
                int(round(selected[1])),
            )
            cv2.circle(
                debug_image,
                selected_pixel,
                7,
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                debug_image,
                f'ID {tag_id} selected {selected_names[tag_id]}',
                (selected_pixel[0] + 10, selected_pixel[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2,
            )

        if not cv2.imwrite(DEBUG_IMAGE_FILE, debug_image):
            raise RuntimeError(
                f'Failed to save debug image: {DEBUG_IMAGE_FILE}'
            )

    # ---------------- 固定拍照姿態，一次拿全部4個標記的像素座標 ----------------
    def move_to_arm_pose(self, arm_pose):
        """直接把 arm_pose 當成法蘭/TCP姿態下PTP，不經過 cam2arm 轉換。
        跟 arm_controller_v2.py 用 armpos 移動的方式一致。"""
        pose = Twist()
        [pose.linear.x, pose.linear.y, pose.linear.z] = arm_pose[0:3]
        [pose.angular.x, pose.angular.y, pose.angular.z] = arm_pose[3:6]
        req = self.generate_robot_request(
            holding=True,
            cmd_mode=RobotCommand.Request.PTP,
            tool=0,
            base=0,
            pose=pose,
        )
        response = self.call_hiwin(req)
        if response is None:
            raise RuntimeError('PTP command failed: no service response.')
        if response.arm_state != RobotCommand.Response.IDLE:
            raise RuntimeError(
                'PTP command did not finish in IDLE state: '
                f'{response.arm_state}'
            )

    def get_current_robot_pose(self):
        request = self.generate_robot_request(
            cmd_mode=RobotCommand.Request.CHECK_POSE,
            tool=0,
            base=0,
        )
        response = self.call_hiwin(request)
        if response is None:
            raise RuntimeError('CHECK_POSE failed: no service response.')

        actual_robot_pose = np.asarray(
            list(response.current_position),
            dtype=np.float64,
        )
        if actual_robot_pose.size < 6:
            raise RuntimeError(
                'CHECK_POSE current_position must contain at least six values.'
            )
        actual_robot_pose = actual_robot_pose[:6]
        if not np.all(np.isfinite(actual_robot_pose)):
            raise ValueError('CHECK_POSE returned non-finite pose values.')
        return actual_robot_pose

    @staticmethod
    def build_base_to_tool_matrix(actual_robot_pose):
        actual_robot_pose = np.asarray(
            actual_robot_pose,
            dtype=np.float64,
        )
        if actual_robot_pose.shape != (6,):
            raise ValueError(
                'Robot pose must be [x, y, z, rx, ry, rz].'
            )
        if not np.all(np.isfinite(actual_robot_pose)):
            raise ValueError('Robot pose contains non-finite values.')

        # Hiwin 舊程式使用 transformations 的 axes='sxyz'，代表
        # static/extrinsic XYZ；在 SciPy Rotation 中對應小寫 'xyz'。
        rotation = Rotation.from_euler(
            'xyz',
            actual_robot_pose[3:6],
            degrees=True,
        ).as_matrix()
        translation_m = actual_robot_pose[:3] / 1000.0
        return make_transform(rotation, translation_m)

    def capture_pixel_corners(self):
        self.move_to_arm_pose(PHOTO_ARM_POSE)
        actual_robot_pose = self.get_current_robot_pose()
        self.get_logger().info(
            f'CHECK_POSE actual pose: {actual_robot_pose.tolist()}'
        )
        time.sleep(0.5)
        img = self.capture_img()
        idmap, corners = self.detect(img)
        self.save_detection_debug_image(img, idmap, corners)
        self.get_logger().info(
            f'ArUco debug image saved to {DEBUG_IMAGE_FILE}'
        )

        detected_ids = sorted(int(tag_id) for tag_id in idmap)
        missing_ids = [
            tag_id for tag_id in TAG_TABLE_POS if tag_id not in idmap
        ]
        if missing_ids:
            raise RuntimeError(
                f'Detected ArUco IDs: {detected_ids}; '
                f'Missing ArUco IDs: {missing_ids}; '
                f'ArUco tags {missing_ids} are not visible from '
                'PHOTO_ARM_POSE'
            )

        img_pot = {}
        for tag_id in TAG_TABLE_POS:
            img_pot[tag_id] = self.selected_corner(
                tag_id,
                corners[idmap[tag_id]],
            )
        return img_pot, actual_robot_pose

    # ---------------- main flow ----------------
    def run(self):
        try:
            img_pot, actual_robot_pose = self.capture_pixel_corners()

            base_to_tool = self.build_base_to_tool_matrix(
                actual_robot_pose
            )
            base_to_camera = base_to_tool @ self.tool_to_camera
            camera_origin_base = base_to_camera[:3, 3]
            table_z_base = (
                camera_origin_base[2]
                - CAMERA_TO_TABLE_NORMAL_DISTANCE_M
            )

            fx = float(self.camera_matrix[0, 0])
            fy = float(self.camera_matrix[1, 1])
            cx = float(self.camera_matrix[0, 2])
            cy = float(self.camera_matrix[1, 2])
            rotation_base_camera = base_to_camera[:3, :3]

            for tag_id in TAG_TABLE_POS:
                raw_pixel = np.asarray(
                    img_pot[tag_id],
                    dtype=np.float64,
                )
                self.config[f'img_pot{tag_id}'] = raw_pixel.tolist()

                undistorted_pixel = self.undistort_pixel(raw_pixel)
                u, v = undistorted_pixel
                ray_camera = np.array(
                    [(u - cx) / fx, (v - cy) / fy, 1.0],
                    dtype=np.float64,
                )
                ray_camera /= np.linalg.norm(ray_camera)

                ray_base = rotation_base_camera @ ray_camera
                ray_base /= np.linalg.norm(ray_base)
                if abs(ray_base[2]) < 1e-9:
                    raise RuntimeError(
                        f'ArUco ID {tag_id} ray is parallel to table plane.'
                    )

                intersection_t = (
                    (table_z_base - camera_origin_base[2])
                    / ray_base[2]
                )
                if intersection_t <= 0.0:
                    raise RuntimeError(
                        f'ArUco ID {tag_id} intersection t must be positive; '
                        f'got {intersection_t}.'
                    )

                point_base = (
                    camera_origin_base + intersection_t * ray_base
                )
                point_base_mm = point_base * 1000.0
                self.config[f'pot{tag_id}'] = [
                    float(point_base_mm[0]),
                    float(point_base_mm[1]),
                    float(point_base_mm[2]),
                ]

                self.get_logger().info(
                    f'ArUco ID {tag_id}: raw_pixel={raw_pixel.tolist()}, '
                    f'undistorted_pixel={undistorted_pixel.tolist()}, '
                    f'Base point [mm]={point_base_mm.tolist()}'
                )

            self.config['armpos'] = [
                float(value) for value in actual_robot_pose
            ]
            self.config['zoff'] = 614.0

            print("\n===== Pixel positions =====")
            for tag_id in TAG_TABLE_POS:
                print(
                    f"img_pot{tag_id}: "
                    f"{self.config[f'img_pot{tag_id}']}"
                )

            print("\n===== Base frame positions =====")
            for tag_id in TAG_TABLE_POS:
                print(f"pot{tag_id}: {self.config[f'pot{tag_id}']}")

            print("\n===== Photo pose =====")
            print(f"armpos: {self.config['armpos']}")

            print("\n===== Camera-to-table height =====")
            print(f"zoff: {self.config['zoff']}")

            print("\n===== YAML format =====")
            print(yaml.dump(
                self.config,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            ))

            self.get_logger().info(
                'Calibration completed. No file was written.'
            )
        except Exception as exc:
            self.get_logger().error(f'Calibration failed: {exc}')
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TableCaliV3()
    thread = threading.Thread(target=node.run, daemon=True)
    thread.start()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
