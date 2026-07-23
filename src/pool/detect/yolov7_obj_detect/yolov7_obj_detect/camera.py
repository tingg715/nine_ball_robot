#!/usr/bin/env python3

import os
import configparser

import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory


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

        self.declare_parameter(
            'calibration_file',
            ''
        )

        self.declare_parameter(
            'show_raw_image',
            True
        )

        self.declare_parameter(
            'show_calibrated_image',
            True
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

        calibration_file_parameter = (
            self.get_parameter('calibration_file')
            .get_parameter_value()
            .string_value
        )

        self.show_raw_image = (
            self.get_parameter('show_raw_image')
            .get_parameter_value()
            .bool_value
        )

        self.show_calibrated_image = (
            self.get_parameter('show_calibrated_image')
            .get_parameter_value()
            .bool_value
        )

        self.alpha = (
            self.get_parameter('alpha')
            .get_parameter_value()
            .double_value
        )

        # ==================================================
        # Find camera calibration file
        # ==================================================
        self.calibration_file = self.find_calibration_file(
            calibration_file_parameter
        )

        # ==================================================
        # Load camera intrinsic and distortion parameters
        # ==================================================
        (
            self.camera_matrix,
            self.distortion_coefficients
        ) = self.load_calibration_file(
            self.calibration_file
        )

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
            f'Calibration file: {self.calibration_file}'
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
            'Press Q or ESC in the OpenCV window to close the display.'
        )

    def find_calibration_file(self, parameter_path):
        """
        搜尋 camera_calibration.ini。

        搜尋順序：
        1. ROS 參數指定的完整路徑
        2. camera.py 同一個資料夾
        3. 套件 share/config 目錄
        4. 套件 share 目錄
        5. 執行指令時所在的目錄
        """

        candidate_paths = []

        # 1. ROS parameter 指定路徑
        if parameter_path:
            candidate_paths.append(
                os.path.abspath(
                    os.path.expanduser(parameter_path)
                )
            )

        # 2. camera.py 同一個資料夾
        script_directory = os.path.dirname(
            os.path.abspath(__file__)
        )

        candidate_paths.append(
            os.path.join(
                script_directory,
                'camera_calibration.ini'
            )
        )

        # 3、4. ROS 2 package share directory
        try:
            package_share_directory = (
                get_package_share_directory(
                    'yolov7_obj_detect'
                )
            )

            candidate_paths.append(
                os.path.join(
                    package_share_directory,
                    'config',
                    'camera_calibration.ini'
                )
            )

            candidate_paths.append(
                os.path.join(
                    package_share_directory,
                    'camera_calibration.ini'
                )
            )

        except Exception as exception:
            self.get_logger().warning(
                'Could not get package share directory: '
                f'{exception}'
            )

        # 5. Current working directory
        candidate_paths.append(
            os.path.join(
                os.getcwd(),
                'camera_calibration.ini'
            )
        )

        for path in candidate_paths:
            if os.path.isfile(path):
                return path

        searched_paths = '\n'.join(candidate_paths)

        raise FileNotFoundError(
            'Cannot find camera_calibration.ini.\n'
            f'Searched paths:\n{searched_paths}'
        )

    def load_calibration_file(self, calibration_file):
        """
        讀取 camera_calibration.ini。

        支援格式：

        [Distortion]
        k1 = ...
        k2 = ...
        t1 = ...
        t2 = ...
        k3 = ...

        [Intrinsic]
        0_0 = ...
        0_1 = ...
        0_2 = ...
        1_0 = ...
        1_1 = ...
        1_2 = ...
        2_0 = ...
        2_1 = ...
        2_2 = ...
        """

        config = configparser.ConfigParser()

        loaded_files = config.read(
            calibration_file
        )

        if not loaded_files:
            raise RuntimeError(
                'Failed to read calibration file: '
                f'{calibration_file}'
            )

        if 'Intrinsic' not in config:
            raise KeyError(
                'Missing [Intrinsic] section in '
                'camera_calibration.ini'
            )

        if 'Distortion' not in config:
            raise KeyError(
                'Missing [Distortion] section in '
                'camera_calibration.ini'
            )

        intrinsic = config['Intrinsic']
        distortion = config['Distortion']

        try:
            # Camera intrinsic matrix
            camera_matrix = np.array(
                [
                    [
                        float(intrinsic['0_0']),
                        float(intrinsic['0_1']),
                        float(intrinsic['0_2'])
                    ],
                    [
                        float(intrinsic['1_0']),
                        float(intrinsic['1_1']),
                        float(intrinsic['1_2'])
                    ],
                    [
                        float(intrinsic['2_0']),
                        float(intrinsic['2_1']),
                        float(intrinsic['2_2'])
                    ]
                ],
                dtype=np.float64
            )

            # Distortion coefficients
            k1 = float(distortion['k1'])
            k2 = float(distortion['k2'])
            t1 = float(distortion['t1'])
            t2 = float(distortion['t2'])
            k3 = float(distortion['k3'])

        except KeyError as exception:
            raise KeyError(
                'Missing calibration parameter: '
                f'{exception}'
            ) from exception

        except ValueError as exception:
            raise ValueError(
                'Calibration parameter is not a valid number: '
                f'{exception}'
            ) from exception

        # OpenCV distortion coefficient order:
        # [k1, k2, p1, p2, k3]
        #
        # 你的 INI 使用 t1、t2，
        # 在 OpenCV 中分別對應 p1、p2。
        distortion_coefficients = np.array(
            [
                k1,
                k2,
                t1,
                t2,
                k3
            ],
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

        self.get_logger().info(
            f'New camera matrix:\n{self.new_camera_matrix}'
        )

        self.get_logger().info(
            f'Valid ROI: {roi}'
        )

    def resize_for_display(
        self,
        image,
        max_width=960
    ):
        """
        只縮放 OpenCV 顯示視窗。

        不會改變發布到 /camera_calib 的影像尺寸。
        """

        height, width = image.shape[:2]

        if width <= max_width:
            return image

        scale = max_width / float(width)

        new_width = int(width * scale)
        new_height = int(height * scale)

        return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    def image_callback(self, message):
        """
        接收 /rgb/image_raw。

        流程：
        1. ROS Image 轉 OpenCV
        2. 去除鏡頭畸變
        3. 發布至 /camera_calib
        4. 顯示原始與校正後影像
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

        # ==================================================
        # Create display images
        # 使用 copy，避免文字被發布出去
        # ==================================================
        # raw_display_image = raw_image.copy()
        # calibrated_display_image = calibrated_image.copy()

        # cv2.putText(
        #     raw_display_image,
        #     'Raw Image',
        #     (30, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 255, 255),
        #     2,
        #     cv2.LINE_AA
        # )

        # cv2.putText(
        #     calibrated_display_image,
        #     'Camera Calibrated Image',
        #     (30, 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 255, 0),
        #     2,
        #     cv2.LINE_AA
        # )

        # raw_display_image = self.resize_for_display(
        #     raw_display_image
        # )

        # calibrated_display_image = self.resize_for_display(
        #     calibrated_display_image
        # )

        # ==================================================
        # Show OpenCV windows
        # ==================================================

        # if self.show_raw_image:
        #     cv2.imshow(
        #         'Raw RGB Image',
        #         raw_display_image
        #     )

        # if self.show_calibrated_image:
        #     cv2.imshow(
        #         'Camera Calibrated Image',
        #         calibrated_display_image
        #     )

        # if (
        #     self.show_raw_image
        #     or self.show_calibrated_image
        # ):
        #     key = cv2.waitKey(1) & 0xFF

        #     if key == ord('q') or key == 27:
        #         self.get_logger().info(
        #             'Closing OpenCV windows.'
        #         )

        #         cv2.destroyAllWindows()

        #         self.show_raw_image = False
        #         self.show_calibrated_image = False

    def destroy_node(self):
        cv2.destroyAllWindows()
        return super().destroy_node()


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
        cv2.destroyAllWindows()

        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()