#!/usr/bin/env python3

"""
Camera calibration node (undistortion)

內參與畸變係數以 ROS 2 parameter 提供,預設值為 Azure Kinect
color_resolution = 1080P (1920x1080) 的實測校正結果。

若更改 driver.launch.py 的 color_resolution,
必須重新校正並覆蓋以下參數,否則座標換算會產生系統性偏差。

本節點不開啟任何顯示視窗,僅訂閱影像、去畸變後重新發布。
如需檢視結果請使用 rqt_image_view 或 rviz2 訂閱 output_topic。
"""

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# ======================================================
# Default intrinsics
# Azure Kinect RGB camera @ 1080P (1920x1080)
# ======================================================

# 3x3 內參矩陣,以 row-major 展平成 9 個元素
# [fx,  0, cx,
#   0, fy, cy,
#   0,  0,  1]
DEFAULT_CAMERA_MATRIX = [
    921.1464532814899, 0.0,              957.208934796569,
    0.0,              919.785424897187, 553.1754854696636,
    0.0,              0.0,               1.0,
]

# OpenCV 畸變係數順序: [k1, k2, p1, p2, k3]
# (原 camera_calibration.ini 中的 t1、t2 即 OpenCV 的 p1、p2)
DEFAULT_DISTORTION_COEFFICIENTS = [
    0.08790006867482206,
    -0.062180853042901774,
    0.0023044976254808424,
    -0.0008184342596589589,
    0.02495957912562247,
]


class CameraCalibNode(Node):

    def __init__(self):
        # ROS 2 節點名稱
        super().__init__('camera_calib')

        # ==================================================
        # ROS 2 parameters
        # ==================================================
        self.declare_parameter(
            'image_topic',
            '/rgb/image_raw'
        )

        self.declare_parameter(
            'output_topic',
            '/camera_calib'
        )

        # 3x3 內參矩陣,row-major 展平的 9 個 double
        self.declare_parameter(
            'camera_matrix',
            DEFAULT_CAMERA_MATRIX
        )

        # 畸變係數 [k1, k2, p1, p2, k3]
        self.declare_parameter(
            'distortion_coefficients',
            DEFAULT_DISTORTION_COEFFICIENTS
        )

        # alpha:
        # 0.0：裁掉黑邊，畫面範圍較小
        # 1.0：保留完整畫面，可能會有黑邊
        self.declare_parameter(
            'alpha',
            1.0
        )

        # ==================================================
        # Read ROS parameters
        # ==================================================
        self.image_topic = (
            self.get_parameter('image_topic')
            .get_parameter_value()
            .string_value
        )

        self.output_topic = (
            self.get_parameter('output_topic')
            .get_parameter_value()
            .string_value
        )

        self.alpha = (
            self.get_parameter('alpha')
            .get_parameter_value()
            .double_value
        )

        # ==================================================
        # Load camera intrinsic and distortion parameters
        # ==================================================
        (
            self.camera_matrix,
            self.distortion_coefficients
        ) = self.load_calibration_parameters()

        # ==================================================
        # CvBridge
        # ==================================================
        self.bridge = CvBridge()

        # ==================================================
        # Cache for undistortion map
        # ==================================================
        self.map_x = None
        self.map_y = None
        self.new_camera_matrix = None
        self.current_image_size = None

        # ==================================================
        # Publisher
        # ==================================================
        self.calibrated_image_publisher = self.create_publisher(
            Image,
            self.output_topic,
            10
        )

        # ==================================================
        # Subscriber
        # ==================================================
        self.image_subscription = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10
        )

        # 避免 Python garbage collector 清除 subscription
        self.image_subscription

        # ==================================================
        # Log information
        # ==================================================
        self.get_logger().info(
            'Camera calibration node started.'
        )

        self.get_logger().info(
            f'Input image topic: {self.image_topic}'
        )

        self.get_logger().info(
            f'Output calibrated image topic: {self.output_topic}'
        )

        self.get_logger().info(
            f'Camera matrix:\n{self.camera_matrix}'
        )

        self.get_logger().info(
            'Distortion coefficients '
            '[k1, k2, p1, p2, k3]: '
            f'{self.distortion_coefficients.flatten().tolist()}'
        )

        self.get_logger().info(
            f'Alpha: {self.alpha}'
        )

    def load_calibration_parameters(self):
        """
        從 ROS 2 parameter 讀取內參與畸變係數。

        camera_matrix:
            9 個 double,row-major 展平的 3x3 矩陣

        distortion_coefficients:
            5 個 double,順序為 [k1, k2, p1, p2, k3]
        """

        camera_matrix_values = list(
            self.get_parameter('camera_matrix')
            .get_parameter_value()
            .double_array_value
        )

        distortion_values = list(
            self.get_parameter('distortion_coefficients')
            .get_parameter_value()
            .double_array_value
        )

        if len(camera_matrix_values) != 9:
            raise ValueError(
                'Parameter "camera_matrix" must contain '
                'exactly 9 values (row-major 3x3), '
                f'but got {len(camera_matrix_values)}.'
            )

        if len(distortion_values) != 5:
            raise ValueError(
                'Parameter "distortion_coefficients" must contain '
                'exactly 5 values [k1, k2, p1, p2, k3], '
                f'but got {len(distortion_values)}.'
            )

        camera_matrix = np.array(
            camera_matrix_values,
            dtype=np.float64
        ).reshape(3, 3)

        distortion_coefficients = np.array(
            distortion_values,
            dtype=np.float64
        ).reshape(1, 5)

        return (
            camera_matrix,
            distortion_coefficients
        )

    def create_undistortion_map(
        self,
        image_width,
        image_height
    ):
        """
        建立去畸變映射表。

        只會在：
        1. 第一次收到影像
        2. 輸入影像解析度改變

        時重新建立。
        """

        image_size = (
            image_width,
            image_height
        )

        if (
            self.current_image_size == image_size
            and self.map_x is not None
            and self.map_y is not None
        ):
            return

        self.current_image_size = image_size

        # 檢查影像解析度與內參是否可能不匹配
        # 主點應該落在畫面中心附近
        expected_cx = image_width / 2.0
        expected_cy = image_height / 2.0

        actual_cx = self.camera_matrix[0, 2]
        actual_cy = self.camera_matrix[1, 2]

        if (
            abs(actual_cx - expected_cx) > image_width * 0.1
            or abs(actual_cy - expected_cy) > image_height * 0.1
        ):
            self.get_logger().warning(
                'Principal point is far from image center. '
                'The intrinsics may not match this resolution. '
                f'Image: {image_width}x{image_height}, '
                f'cx={actual_cx:.2f} (expected ~{expected_cx:.1f}), '
                f'cy={actual_cy:.2f} (expected ~{expected_cy:.1f})'
            )

        self.new_camera_matrix, roi = (
            cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.distortion_coefficients,
                image_size,
                self.alpha,
                image_size
            )
        )

        self.map_x, self.map_y = (
            cv2.initUndistortRectifyMap(
                self.camera_matrix,
                self.distortion_coefficients,
                None,
                self.new_camera_matrix,
                image_size,
                cv2.CV_32FC1
            )
        )

        self.get_logger().info(
            'Created undistortion map for image size: '
            f'{image_width} x {image_height}'
        )

        # 下游若使用 output_topic 的影像做像素→世界座標轉換,
        # 必須使用這組 new camera matrix,而非原始內參。
        self.get_logger().info(
            'New camera matrix (use this for undistorted images):'
            f'\n{self.new_camera_matrix}'
        )

        self.get_logger().info(
            f'Valid ROI: {roi}'
        )

    def image_callback(self, message):
        """
        接收輸入影像。

        流程：
        1. ROS Image 轉 OpenCV
        2. 去除鏡頭畸變
        3. 發布至 output_topic
        """

        # ==================================================
        # Convert ROS Image to OpenCV image
        # ==================================================
        try:
            raw_image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding='bgr8'
            )

        except Exception as exception:
            self.get_logger().error(
                'Failed to convert ROS Image '
                f'to OpenCV image: {exception}'
            )
            return

        if raw_image is None:
            self.get_logger().error(
                'Received empty image.'
            )
            return

        image_height, image_width = raw_image.shape[:2]

        # ==================================================
        # Undistort image
        # ==================================================
        try:
            self.create_undistortion_map(
                image_width,
                image_height
            )

            calibrated_image = cv2.remap(
                raw_image,
                self.map_x,
                self.map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

        except cv2.error as exception:
            self.get_logger().error(
                'OpenCV undistortion failed: '
                f'{exception}'
            )
            return

        # ==================================================
        # Publish calibrated image
        # ==================================================
        try:
            calibrated_message = self.bridge.cv2_to_imgmsg(
                calibrated_image,
                encoding='bgr8'
            )

            # 保留原始相機影像的時間戳記與 frame_id
            calibrated_message.header = message.header

            self.calibrated_image_publisher.publish(
                calibrated_message
            )

        except Exception as exception:
            self.get_logger().error(
                'Failed to publish calibrated image: '
                f'{exception}'
            )
            return


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = CameraCalibNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as exception:
        print(
            f'Camera calibration node error: {exception}'
        )

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
