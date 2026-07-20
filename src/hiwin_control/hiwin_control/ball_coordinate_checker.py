#!/usr/bin/env python3
import ast
import cv2
import configparser
import math
import os
from typing import List, Optional

import numpy as np
import quaternion as qtn
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

import hiwin_control.transformations as transformations


class BallCoordinateChecker(Node):
    """
    Subscribe to YOLO ball pixel coordinates and convert every ball position to:
      1. camera-relative millimetres
      2. Hiwin base-frame millimetres

    Assumptions:
      - The robot/camera is already at FIX_ABS_CAM (arm.yaml -> armpos).
      - center_data_coords is [u1, v1, u2, v2, ...].
      - center_data_labels has the same ball ordering as center_data_coords.
    """

    def __init__(self) -> None:
        super().__init__('ball_coordinate_checker')

        # ---------- ROS parameters ----------
        default_workspace = os.getcwd()
        self.declare_parameter(
            'arm_yaml',
            os.path.join(
                default_workspace,
                'src/hiwin_control/hiwin_control/arm.yaml',
            ),
        )
        self.declare_parameter(
            'hand_eye_ini',
            os.path.join(
                default_workspace,
                'src/hiwin_control/hiwin_control/eye_in_hand_calibration.ini',
            ),
        )
        self.declare_parameter(
            'camera_calibration_ini',
            os.path.join(
                default_workspace,
                'src/hiwin_control/hiwin_control/camera_calibration.ini',
            ),
        )

        # Camera intrinsic parameters are loaded from camera_calibration.ini.
        # These are preferable to using only camera FOV.
        self.declare_parameter('fx', 923.2900734226928)
        self.declare_parameter('fy', 922.8913296403708)
        self.declare_parameter('cx', 956.336780370788)
        self.declare_parameter('cy', 547.7923321648962)

        self.arm_yaml_path = (
            self.get_parameter('arm_yaml').get_parameter_value().string_value
        )
        self.hand_eye_ini_path = (
            self.get_parameter('hand_eye_ini').get_parameter_value().string_value
        )
        self.camera_calibration_ini_path = (
            self.get_parameter('camera_calibration_ini')
            .get_parameter_value().string_value
        )

        self.fx = self.get_parameter('fx').value
        self.fy = self.get_parameter('fy').value
        self.cx = self.get_parameter('cx').value
        self.cy = self.get_parameter('cy').value

        # ---------- Read calibration data ----------
        self.fixed_camera_pose, self.camera_to_table = self._load_arm_yaml(
            self.arm_yaml_path
        )

        (
            self.tool_to_camera_translation,
            self.tool_to_camera_quaternion,
        ) = self._load_hand_eye_calibration(self.hand_eye_ini_path)

        (
            self.camera_matrix,
            self.dist_coeffs,
        ) = self._load_camera_calibration(self.camera_calibration_ini_path)

        self.fx = float(self.camera_matrix[0, 0])
        self.fy = float(self.camera_matrix[1, 1])
        self.cx = float(self.camera_matrix[0, 2])
        self.cy = float(self.camera_matrix[1, 2])

        # Latest labels from center_data_labels.
        self.labels: List[str] = []

        self.create_subscription(
            String,
            'center_data_labels',
            self.label_callback,
            10,
        )
        self.create_subscription(
            Float64MultiArray,
            'center_data_coords',
            self.coordinate_callback,
            10,
        )

        self.get_logger().info('Ball coordinate checker started.')
        self.get_logger().info(
            f'Fixed camera pose: {self.fixed_camera_pose}'
        )
        self.get_logger().info(
            f'Camera-to-table height: {self.camera_to_table:.3f} mm'
        )
        self.get_logger().info(
            'Tool-to-Camera translation [x, y, z] m: '
            f'{self.tool_to_camera_translation.tolist()}'
        )
        self.get_logger().info(
            'Tool-to-Camera quaternion [qx, qy, qz, qw]: '
            f'{self.tool_to_camera_quaternion.tolist()}'
        )
        self.get_logger().info(
            f'Camera intrinsic: fx={self.fx:.6f}, fy={self.fy:.6f}, '
            f'cx={self.cx:.6f}, cy={self.cy:.6f}'
        )
        self.get_logger().info(
            f'Distortion coefficients: {self.dist_coeffs.tolist()}'
        )

    def _load_arm_yaml(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'arm.yaml not found: {path}')

        with open(path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)

        if 'armpos' not in data or 'zoff' not in data:
            raise KeyError('arm.yaml must contain armpos and zoff.')

        fixed_camera_pose = [float(value) for value in data['armpos']]
        camera_to_table = float(data['zoff'])

        if len(fixed_camera_pose) != 6:
            raise ValueError('armpos must contain [x, y, z, rx, ry, rz].')

        return fixed_camera_pose, camera_to_table

    def _load_hand_eye_calibration(self, path: str):
        """
        Load the Tool-to-Camera transform from an INI file.

        Translation is in metres and quaternion order is [qx, qy, qz, qw].
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'eye_in_hand_calibration.ini not found: {path}'
            )

        config = configparser.ConfigParser()
        loaded_files = config.read(path)
        if not loaded_files:
            raise ValueError(
                f'Unable to read eye-in-hand calibration INI: {path}'
            )

        if 'hand_eye_calibration' not in config:
            raise KeyError(
                'eye_in_hand_calibration.ini has no '
                '[hand_eye_calibration] section.'
            )

        calibration = config['hand_eye_calibration']
        translation_fields = ('x', 'y', 'z')
        quaternion_fields = ('qx', 'qy', 'qz', 'qw')
        required_fields = translation_fields + quaternion_fields
        missing_fields = [
            field for field in required_fields if field not in calibration
        ]
        if missing_fields:
            raise KeyError(
                'Missing field(s) in [hand_eye_calibration]: '
                + ', '.join(missing_fields)
            )

        try:
            translation_values = [
                float(calibration[field]) for field in translation_fields
            ]
            quaternion_values = [
                float(calibration[field]) for field in quaternion_fields
            ]
        except ValueError as error:
            raise ValueError(
                'All [hand_eye_calibration] fields '
                'x, y, z, qx, qy, qz, qw must be valid floats.'
            ) from error

        tool_to_camera_translation = np.array(
            translation_values,
            dtype=np.float64,
        )

        tool_to_camera_quaternion = np.array(
            quaternion_values,
            dtype=np.float64,
        )
        quaternion_norm = float(np.linalg.norm(tool_to_camera_quaternion))
        if quaternion_norm == 0.0:
            raise ValueError(
                'Tool-to-Camera quaternion norm is 0; '
                'qx, qy, qz, qw must define a valid rotation.'
            )
        tool_to_camera_quaternion /= quaternion_norm

        return tool_to_camera_translation, tool_to_camera_quaternion

    def _load_camera_calibration(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f'camera_calibration.ini not found: {path}'
            )

        config = configparser.ConfigParser()
        config.read(path)

        if 'Intrinsic' not in config:
            raise KeyError(
                'camera_calibration.ini has no [Intrinsic] section.'
            )
        if 'Distortion' not in config:
            raise KeyError(
                'camera_calibration.ini has no [Distortion] section.'
            )

        intrinsic = config['Intrinsic']
        distortion = config['Distortion']

        camera_matrix = np.array([
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
        ], dtype=np.float64)

        # OpenCV distortion order: [k1, k2, p1, p2, k3]
        # Your file uses t1/t2, which correspond to p1/p2.
        dist_coeffs = np.array([
            float(distortion['k1']),
            float(distortion['k2']),
            float(distortion['t1']),
            float(distortion['t2']),
            float(distortion['k3']),
        ], dtype=np.float64)

        return camera_matrix, dist_coeffs

    def label_callback(self, msg: String) -> None:
        try:
            parsed = ast.literal_eval(msg.data)
            if isinstance(parsed, (list, tuple)):
                self.labels = [str(label) for label in parsed]
            else:
                self.get_logger().warning(
                    'center_data_labels is not a list or tuple.'
                )
        except (ValueError, SyntaxError) as error:
            self.get_logger().warning(
                f'Unable to parse center_data_labels: {error}'
            )

    def coordinate_callback(self, msg: Float64MultiArray) -> None:
        coordinates = list(msg.data)

        if not coordinates:
            self.get_logger().warning('Received empty coordinate data.')
            return

        if len(coordinates) % 2 != 0:
            self.get_logger().error(
                f'Coordinate count must be even, got {len(coordinates)}.'
            )
            return

        ball_pixels = np.asarray(
            coordinates,
            dtype=np.float64,
        ).reshape(-1, 2)

        self.get_logger().info(
            f'========== Detected {len(ball_pixels)} balls =========='
        )

        for index, pixel in enumerate(ball_pixels):
            label = (
                self.labels[index]
                if index < len(self.labels)
                else f'unknown_{index}'
            )

            corrected_pixel = self.undistort_pixel(
                pixel_u=float(pixel[0]),
                pixel_v=float(pixel[1]),
            )

            camera_point_mm = self.pixel_to_camera_mm(
                pixel_u=float(corrected_pixel[0]),
                pixel_v=float(corrected_pixel[1]),
                camera_to_table_mm=self.camera_to_table,
            )

            base_point_mm = self.camera_point_to_base_mm(
                camera_point_mm=camera_point_mm,
                robot_pose=self.fixed_camera_pose,
            )

            self.get_logger().info(
                '\n'
                f'Ball {index + 1}: {label}\n'
                f'  raw pixel (u, v)       = '
                f'({pixel[0]:.2f}, {pixel[1]:.2f})\n'
                f'  corrected pixel (u, v) = '
                f'({corrected_pixel[0]:.2f}, {corrected_pixel[1]:.2f})\n'
                f'  camera (x, y, z) mm    = '
                f'({camera_point_mm[0]:.2f}, '
                f'{camera_point_mm[1]:.2f}, '
                f'{camera_point_mm[2]:.2f})\n'
                f'  base   (x, y, z) mm = '
                f'({base_point_mm[0]:.2f}, '
                f'{base_point_mm[1]:.2f}, '
                f'{base_point_mm[2]:.2f})'
            )

    def undistort_pixel(
        self,
        pixel_u: float,
        pixel_v: float,
    ) -> np.ndarray:
        point = np.array(
            [[[pixel_u, pixel_v]]],
            dtype=np.float64,
        )

        corrected = cv2.undistortPoints(
            point,
            self.camera_matrix,
            self.dist_coeffs,
            P=self.camera_matrix,
        )

        return corrected[0, 0]

    def pixel_to_camera_mm(
        self,
        pixel_u: float,
        pixel_v: float,
        camera_to_table_mm: float,
    ) -> np.ndarray:
        """
        Pinhole-camera conversion:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            Z = camera-to-table distance

        Returns homogeneous [x_mm, y_mm, z_mm, 1].
        """
        x_mm = (pixel_u - self.cx) * camera_to_table_mm / self.fx
        y_mm = (pixel_v - self.cy) * camera_to_table_mm / self.fy
        z_mm = camera_to_table_mm

        return np.array(
            [x_mm, y_mm, z_mm, 1.0],
            dtype=np.float64,
        )

    def camera_point_to_base_mm(
        self,
        camera_point_mm: np.ndarray,
        robot_pose: List[float],
    ) -> np.ndarray:
        """
        Transform:
            base_T_ball = base_T_tool @ tool_T_camera @ camera_T_ball
        """
        base_to_tool = self.pose_to_matrix(robot_pose)

        qx, qy, qz, qw = self.tool_to_camera_quaternion
        tool_to_camera_rotation = qtn.as_rotation_matrix(
            np.quaternion(qw, qx, qy, qz)
        )

        tool_to_camera = np.eye(4, dtype=np.float64)
        tool_to_camera[:3, :3] = tool_to_camera_rotation
        tool_to_camera[:3, 3] = self.tool_to_camera_translation

        camera_point_m = camera_point_mm.copy()
        camera_point_m[:3] /= 1000.0

        base_point_m = (
            base_to_tool
            @ tool_to_camera
            @ camera_point_m
        )

        return base_point_m[:3] * 1000.0

    @staticmethod
    def pose_to_matrix(robot_pose: List[float]) -> np.ndarray:
        """
        robot_pose = [x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg]
        """
        pose = np.asarray(robot_pose, dtype=np.float64).copy()
        translation_m = pose[:3] / 1000.0

        roll, pitch, yaw = np.deg2rad(pose[3:6])

        quaternion = transformations.quaternion_from_euler(
            roll,
            pitch,
            yaw,
            axes='sxyz',
        )

        rotation = qtn.as_rotation_matrix(
            np.quaternion(
                quaternion[3],
                quaternion[0],
                quaternion[1],
                quaternion[2],
            )
        )

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation_m

        return matrix


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[BallCoordinateChecker] = None

    try:
        node = BallCoordinateChecker()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'Fatal error: {error}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
