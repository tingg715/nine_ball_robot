#!/usr/bin/env python3

import ast
import configparser
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import quaternion as qtn
import rclpy
import yaml

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

import hiwin_control.transformations as transformations


class BallCoordinateChecker(Node):
    """
    接收 YOLO 球中心像素座標，轉換成機械手臂 Base 座標，
    並將球標籤與 Base XYZ 一起發布成 JSON。

    訂閱：
      /center_data_coords
        型別：std_msgs/msg/Float64MultiArray
        格式：[u1, v1, u2, v2, ...]

      /center_data_labels
        型別：std_msgs/msg/String
        格式："['1', '5', 'white']"

    發布：
      /ball_coordinate
        型別：std_msgs/msg/String

        JSON 格式：
        [
          {"label": "1", "x": 100.0, "y": 200.0, "z": -120.0},
          {"label": "white", "x": 300.0, "y": 400.0, "z": -120.0}
        ]

    前提：
      1. YOLO 使用已完成去畸變的 /camera_calib 影像。
      2. 手臂拍照時位於 arm.yaml 的 armpos。
      3. arm.yaml 的 zoff 是相機到桌面的距離，單位為 mm。
      4. eye_in_hand_calibration.ini 儲存 Tool-to-Camera 轉換。
      5. camera.py 和本程式使用相同解析度及 alpha。
    """

    def __init__(self) -> None:
        super().__init__('ball_coordinate_checker')

        workspace = os.getcwd()

        # ==================================================
        # ROS parameters
        # ==================================================
        self.declare_parameter(
            'arm_yaml',
            os.path.join(
                workspace,
                'src/hiwin_control/hiwin_control/arm.yaml',
            ),
        )

        self.declare_parameter(
            'hand_eye_ini',
            os.path.join(
                workspace,
                'src/hiwin_control/hiwin_control/'
                'eye_in_hand_calibration.ini',
            ),
        )

        self.declare_parameter(
            'camera_calibration_ini',
            os.path.join(
                workspace,
                'src/hiwin_control/hiwin_control/'
                'camera_calibration.ini',
            ),
        )

        # 必須和 camera.py 的影像解析度一致
        self.declare_parameter('image_width', 1920)
        self.declare_parameter('image_height', 1080)

        # 必須和 camera.py 的 alpha 一致
        self.declare_parameter('alpha', 1.0)

        self.declare_parameter(
            'coordinate_topic',
            '/center_data_coords',
        )

        self.declare_parameter(
            'label_topic',
            '/center_data_labels',
        )

        self.declare_parameter(
            'output_topic',
            '/ball_coordinate',
        )

        # ==================================================
        # Read ROS parameters
        # ==================================================
        self.arm_yaml_path = (
            self.get_parameter('arm_yaml')
            .get_parameter_value()
            .string_value
        )

        self.hand_eye_ini_path = (
            self.get_parameter('hand_eye_ini')
            .get_parameter_value()
            .string_value
        )

        self.camera_calibration_ini_path = (
            self.get_parameter('camera_calibration_ini')
            .get_parameter_value()
            .string_value
        )

        self.image_width = int(
            self.get_parameter('image_width').value
        )

        self.image_height = int(
            self.get_parameter('image_height').value
        )

        self.alpha = float(
            self.get_parameter('alpha').value
        )

        self.coordinate_topic = (
            self.get_parameter('coordinate_topic')
            .get_parameter_value()
            .string_value
        )

        self.label_topic = (
            self.get_parameter('label_topic')
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter('output_topic')
            .get_parameter_value()
            .string_value
        )

        # ==================================================
        # Load arm photo pose and camera-to-table distance
        # ==================================================
        (
            self.fixed_camera_pose,
            self.camera_to_table_mm,
        ) = self.load_arm_yaml(
            self.arm_yaml_path
        )

        # ==================================================
        # Load Tool-to-Camera transform
        # ==================================================
        (
            self.tool_to_camera_translation_m,
            self.tool_to_camera_quaternion,
        ) = self.load_hand_eye_calibration(
            self.hand_eye_ini_path
        )

        # ==================================================
        # Load original camera calibration
        # ==================================================
        (
            self.original_camera_matrix,
            self.distortion_coefficients,
        ) = self.load_camera_calibration(
            self.camera_calibration_ini_path
        )

        # /camera_calib 使用 getOptimalNewCameraMatrix，
        # 因此像素座標轉換也要使用相同的新內參。
        self.calibrated_camera_matrix = (
            self.calculate_calibrated_camera_matrix()
        )

        self.fx = float(
            self.calibrated_camera_matrix[0, 0]
        )

        self.fy = float(
            self.calibrated_camera_matrix[1, 1]
        )

        self.cx = float(
            self.calibrated_camera_matrix[0, 2]
        )

        self.cy = float(
            self.calibrated_camera_matrix[1, 2]
        )

        # 最新一筆球標籤
        self.labels: List[str] = []

        # 是否已收到有效標籤
        self.labels_received = False

        # ==================================================
        # ROS publisher
        # ==================================================
        self.ball_coordinate_publisher = self.create_publisher(
            String,
            self.output_topic,
            10,
        )

        # ==================================================
        # ROS subscriptions
        # ==================================================
        self.label_subscription = self.create_subscription(
            String,
            self.label_topic,
            self.label_callback,
            10,
        )

        self.coordinate_subscription = self.create_subscription(
            Float64MultiArray,
            self.coordinate_topic,
            self.coordinate_callback,
            10,
        )

        # ==================================================
        # Logs
        # ==================================================
        self.get_logger().info(
            'Ball coordinate checker started.'
        )

        self.get_logger().info(
            f'Coordinate topic: {self.coordinate_topic}'
        )

        self.get_logger().info(
            f'Label topic: {self.label_topic}'
        )

        self.get_logger().info(
            f'JSON output topic: {self.output_topic}'
        )

        self.get_logger().info(
            f'Fixed camera pose: {self.fixed_camera_pose}'
        )

        self.get_logger().info(
            'Camera-to-table distance: '
            f'{self.camera_to_table_mm:.3f} mm'
        )

        self.get_logger().info(
            'Tool-to-Camera translation [m]: '
            f'{self.tool_to_camera_translation_m.tolist()}'
        )

        self.get_logger().info(
            'Tool-to-Camera quaternion [qx, qy, qz, qw]: '
            f'{self.tool_to_camera_quaternion.tolist()}'
        )

        self.get_logger().info(
            'Calibrated camera matrix:\n'
            f'{self.calibrated_camera_matrix}'
        )

        self.get_logger().info(
            f'fx={self.fx:.6f}, '
            f'fy={self.fy:.6f}, '
            f'cx={self.cx:.6f}, '
            f'cy={self.cy:.6f}'
        )

    # ======================================================
    # Load arm.yaml
    # ======================================================
    def load_arm_yaml(
        self,
        path: str,
    ) -> Tuple[List[float], float]:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'arm.yaml not found: {path}'
            )

        with open(
            path,
            'r',
            encoding='utf-8',
        ) as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f'arm.yaml content is invalid: {path}'
            )

        if 'armpos' not in data:
            raise KeyError(
                'arm.yaml does not contain armpos.'
            )

        if 'zoff' not in data:
            raise KeyError(
                'arm.yaml does not contain zoff.'
            )

        fixed_camera_pose = [
            float(value)
            for value in data['armpos']
        ]

        camera_to_table_mm = float(
            data['zoff']
        )

        if len(fixed_camera_pose) != 6:
            raise ValueError(
                'armpos must contain '
                '[x, y, z, rx, ry, rz].'
            )

        if camera_to_table_mm <= 0.0:
            raise ValueError(
                'zoff must be greater than 0.'
            )

        return (
            fixed_camera_pose,
            camera_to_table_mm,
        )

    # ======================================================
    # Load eye-in-hand calibration
    # ======================================================
    def load_hand_eye_calibration(
        self,
        path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                'eye_in_hand_calibration.ini '
                f'not found: {path}'
            )

        config = configparser.ConfigParser()

        loaded_files = config.read(path)

        if not loaded_files:
            raise RuntimeError(
                'Unable to read eye-in-hand calibration file: '
                f'{path}'
            )

        section_name = 'hand_eye_calibration'

        if section_name not in config:
            raise KeyError(
                'eye_in_hand_calibration.ini does not contain '
                '[hand_eye_calibration].'
            )

        calibration = config[section_name]

        required_fields = (
            'x',
            'y',
            'z',
            'qx',
            'qy',
            'qz',
            'qw',
        )

        missing_fields = [
            field
            for field in required_fields
            if field not in calibration
        ]

        if missing_fields:
            raise KeyError(
                'Missing eye-in-hand field(s): '
                + ', '.join(missing_fields)
            )

        try:
            translation_m = np.array(
                [
                    float(calibration['x']),
                    float(calibration['y']),
                    float(calibration['z']),
                ],
                dtype=np.float64,
            )

            quaternion = np.array(
                [
                    float(calibration['qx']),
                    float(calibration['qy']),
                    float(calibration['qz']),
                    float(calibration['qw']),
                ],
                dtype=np.float64,
            )

        except ValueError as error:
            raise ValueError(
                'Eye-in-hand parameters must be valid floats.'
            ) from error

        quaternion_norm = float(
            np.linalg.norm(quaternion)
        )

        if quaternion_norm == 0.0:
            raise ValueError(
                'Eye-in-hand quaternion norm is zero.'
            )

        quaternion /= quaternion_norm

        return (
            translation_m,
            quaternion,
        )

    # ======================================================
    # Load camera calibration
    # ======================================================
    def load_camera_calibration(
        self,
        path: str,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not os.path.isfile(path):
            raise FileNotFoundError(
                'camera_calibration.ini '
                f'not found: {path}'
            )

        config = configparser.ConfigParser()

        loaded_files = config.read(path)

        if not loaded_files:
            raise RuntimeError(
                'Unable to read camera calibration file: '
                f'{path}'
            )

        if 'Intrinsic' not in config:
            raise KeyError(
                'camera_calibration.ini has no '
                '[Intrinsic] section.'
            )

        if 'Distortion' not in config:
            raise KeyError(
                'camera_calibration.ini has no '
                '[Distortion] section.'
            )

        intrinsic = config['Intrinsic']
        distortion = config['Distortion']

        try:
            camera_matrix = np.array(
                [
                    [
                        float(intrinsic['0_0']),
                        float(intrinsic['0_1']),
                        float(intrinsic['0_2']),
                    ],
                    [
                        float(intrinsic['1_0']),
                        float(intrinsic['1_1']),
                        float(intrinsic['1_2']),
                    ],
                    [
                        float(intrinsic['2_0']),
                        float(intrinsic['2_1']),
                        float(intrinsic['2_2']),
                    ],
                ],
                dtype=np.float64,
            )

            # OpenCV 的順序是：
            # [k1, k2, p1, p2, k3]
            #
            # INI 中的 t1、t2 對應 p1、p2。
            distortion_coefficients = np.array(
                [
                    float(distortion['k1']),
                    float(distortion['k2']),
                    float(distortion['t1']),
                    float(distortion['t2']),
                    float(distortion['k3']),
                ],
                dtype=np.float64,
            )

        except (KeyError, ValueError) as error:
            raise ValueError(
                'Invalid camera calibration value: '
                f'{error}'
            ) from error

        return (
            camera_matrix,
            distortion_coefficients,
        )

    # ======================================================
    # Calculate intrinsic used by /camera_calib
    # ======================================================
    def calculate_calibrated_camera_matrix(
        self,
    ) -> np.ndarray:
        """
        必須與 camera.py 的設定完全一致。
        """

        image_size = (
            self.image_width,
            self.image_height,
        )

        new_camera_matrix, _ = (
            cv2.getOptimalNewCameraMatrix(
                self.original_camera_matrix,
                self.distortion_coefficients,
                image_size,
                self.alpha,
                image_size,
            )
        )

        return new_camera_matrix

    # ======================================================
    # Label callback
    # ======================================================
    def label_callback(
        self,
        msg: String,
    ) -> None:

        try:
            parsed = ast.literal_eval(
                msg.data
            )

            if not isinstance(
                parsed,
                (list, tuple),
            ):
                self.get_logger().warning(
                    'center_data_labels is not '
                    'a list or tuple.'
                )
                return

            self.labels = [
                str(label)
                for label in parsed
            ]

            self.labels_received = True

        except (
            ValueError,
            SyntaxError,
        ) as error:
            self.get_logger().warning(
                'Unable to parse center_data_labels: '
                f'{error}'
            )

    # ======================================================
    # Coordinate callback
    # ======================================================
    def coordinate_callback(
        self,
        msg: Float64MultiArray,
    ) -> None:

        coordinates = list(
            msg.data
        )

        if not coordinates:
            self.get_logger().warning(
                'Received empty coordinate data.'
            )
            return

        if len(coordinates) % 2 != 0:
            self.get_logger().error(
                'Coordinate count must be even, '
                f'got {len(coordinates)}.'
            )
            return

        ball_pixels = np.asarray(
            coordinates,
            dtype=np.float64,
        ).reshape(-1, 2)

        if not self.labels_received:
            self.get_logger().warning(
                'Coordinates received, but no labels have been '
                'received yet. Waiting for center_data_labels.'
            )
            return

        if len(self.labels) != len(ball_pixels):
            self.get_logger().warning(
                'Label count and coordinate count do not match. '
                f'Labels: {len(self.labels)}, '
                f'balls: {len(ball_pixels)}. '
                'This frame will not be published.'
            )
            return

        self.get_logger().info(
            '========== '
            f'Detected {len(ball_pixels)} balls '
            '=========='
        )

        # 每一顆球的輸出資料：
        #
        # {
        #   "label": "1",
        #   "x": 100.0,
        #   "y": 200.0,
        #   "z": -120.0
        # }
        ball_coordinate_output = []

        for index, pixel in enumerate(
            ball_pixels
        ):
            label = self.labels[index]

            pixel_u = float(pixel[0])
            pixel_v = float(pixel[1])

            # YOLO 使用已經去畸變的 /camera_calib，
            # 所以不再執行 cv2.undistortPoints。
            camera_point_mm = self.pixel_to_camera_mm(
                pixel_u=pixel_u,
                pixel_v=pixel_v,
                camera_to_table_mm=self.camera_to_table_mm,
            )

            base_point_mm = self.camera_point_to_base_mm(
                camera_point_mm=camera_point_mm,
                robot_pose=self.fixed_camera_pose,
            )

            ball_data = {
                'label': label,
                'x': float(base_point_mm[0]),
                'y': float(base_point_mm[1]),
                'z': float(base_point_mm[2]),
            }

            ball_coordinate_output.append(
                ball_data
            )

            self.get_logger().info(
                '\n'
                f'Ball {index + 1}: {label}\n'
                f'  pixel (u, v) = '
                f'({pixel_u:.2f}, {pixel_v:.2f})\n'
                f'  camera (x, y, z) mm = '
                f'({camera_point_mm[0]:.2f}, '
                f'{camera_point_mm[1]:.2f}, '
                f'{camera_point_mm[2]:.2f})\n'
                f'  base (x, y, z) mm = '
                f'({base_point_mm[0]:.2f}, '
                f'{base_point_mm[1]:.2f}, '
                f'{base_point_mm[2]:.2f})'
            )

        # ==================================================
        # Convert result to JSON and publish
        # ==================================================
        output_message = String()

        output_message.data = json.dumps(
            ball_coordinate_output,
            ensure_ascii=False,
            separators=(',', ':'),
        )

        self.ball_coordinate_publisher.publish(
            output_message
        )

        self.get_logger().info(
            f'Published {len(ball_coordinate_output)} balls '
            f'to {self.output_topic}.'
        )

    # ======================================================
    # Pixel to camera coordinates
    # ======================================================
    def pixel_to_camera_mm(
        self,
        pixel_u: float,
        pixel_v: float,
        camera_to_table_mm: float,
    ) -> np.ndarray:
        """
        校正後影像的針孔相機轉換：

          X = (u - cx) * Z / fx
          Y = (v - cy) * Z / fy
          Z = camera-to-table distance

        回傳齊次座標：
          [x_mm, y_mm, z_mm, 1]
        """

        x_mm = (
            (pixel_u - self.cx)
            * camera_to_table_mm
            / self.fx
        )

        y_mm = (
            (pixel_v - self.cy)
            * camera_to_table_mm
            / self.fy
        )

        z_mm = camera_to_table_mm

        return np.array(
            [
                x_mm,
                y_mm,
                z_mm,
                1.0,
            ],
            dtype=np.float64,
        )

    # ======================================================
    # Camera point to robot Base coordinates
    # ======================================================
    def camera_point_to_base_mm(
        self,
        camera_point_mm: np.ndarray,
        robot_pose: List[float],
    ) -> np.ndarray:
        """
        Base_P_Ball
          = Base_T_Tool
          @ Tool_T_Camera
          @ Camera_P_Ball
        """

        base_to_tool = self.pose_to_matrix(
            robot_pose
        )

        qx, qy, qz, qw = (
            self.tool_to_camera_quaternion
        )

        tool_to_camera_rotation = (
            qtn.as_rotation_matrix(
                np.quaternion(
                    qw,
                    qx,
                    qy,
                    qz,
                )
            )
        )

        tool_to_camera = np.eye(
            4,
            dtype=np.float64,
        )

        tool_to_camera[:3, :3] = (
            tool_to_camera_rotation
        )

        tool_to_camera[:3, 3] = (
            self.tool_to_camera_translation_m
        )

        camera_point_m = (
            camera_point_mm.copy()
        )

        camera_point_m[:3] /= 1000.0

        base_point_m = (
            base_to_tool
            @ tool_to_camera
            @ camera_point_m
        )

        return (
            base_point_m[:3]
            * 1000.0
        )

    # ======================================================
    # Robot pose to homogeneous matrix
    # ======================================================
    @staticmethod
    def pose_to_matrix(
        robot_pose: List[float],
    ) -> np.ndarray:
        """
        robot_pose：
          [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        """

        pose = np.asarray(
            robot_pose,
            dtype=np.float64,
        ).copy()

        translation_m = (
            pose[:3] / 1000.0
        )

        roll, pitch, yaw = np.deg2rad(
            pose[3:6]
        )

        quaternion = (
            transformations.quaternion_from_euler(
                roll,
                pitch,
                yaw,
                axes='sxyz',
            )
        )

        rotation = qtn.as_rotation_matrix(
            np.quaternion(
                quaternion[3],
                quaternion[0],
                quaternion[1],
                quaternion[2],
            )
        )

        matrix = np.eye(
            4,
            dtype=np.float64,
        )

        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation_m

        return matrix


def main(
    args: Optional[List[str]] = None,
) -> None:

    rclpy.init(args=args)

    node: Optional[
        BallCoordinateChecker
    ] = None

    try:
        node = BallCoordinateChecker()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().fatal(
                str(error)
            )
        else:
            print(
                f'Fatal error: {error}'
            )

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()