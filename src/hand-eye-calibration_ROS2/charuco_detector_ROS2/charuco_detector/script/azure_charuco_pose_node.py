#!/usr/bin/env python3

import math
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped

from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster


class AzureCharucoPoseNode(Node):
    def __init__(self):
        super().__init__("azure_charuco_pose_node")

        # ===== ROS parameters =====
        self.declare_parameter("image_topic", "/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/rgb/camera_info")

        self.declare_parameter("parent_frame", "rgb_camera_link")
        self.declare_parameter("charuco_frame", "calib_charuco")

        # 依照你之前的板子設定
        self.declare_parameter("squares_x", 10)
        self.declare_parameter("squares_y", 14)
        self.declare_parameter("square_length", 0.0200)   # m
        self.declare_parameter("marker_length", 0.0150)   # m

        # OpenCV ArUco dictionary
        # DICT_6X6_250 通常是 10
        self.declare_parameter("dictionary_name", "DICT_6X6_250")

        self.declare_parameter("publish_tf", True)
        self.declare_parameter("use_clahe", True)
        self.declare_parameter("show_debug_image", True)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.parent_frame = self.get_parameter("parent_frame").value
        self.charuco_frame = self.get_parameter("charuco_frame").value

        self.squares_x = int(self.get_parameter("squares_x").value)
        self.squares_y = int(self.get_parameter("squares_y").value)
        self.square_length = float(self.get_parameter("square_length").value)
        self.marker_length = float(self.get_parameter("marker_length").value)
        self.dictionary_name = self.get_parameter("dictionary_name").value

        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.use_clahe = bool(self.get_parameter("use_clahe").value)
        self.show_debug_image = bool(self.get_parameter("show_debug_image").value)

        self.bridge = CvBridge()

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False

        # ===== ArUco / ChArUco setup =====
        self.dictionary = self.get_dictionary(self.dictionary_name)
        self.board = self.create_charuco_board()

        self.detector_params = cv2.aruco.DetectorParameters()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_pub = self.create_publisher(PoseStamped, "/charuco/pose", 10)
        self.debug_image_pub = self.create_publisher(Image, "/charuco/debug_image", 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            sensor_qos
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10
        )

        self.get_logger().info("Azure ChArUco Pose Node started")
        self.get_logger().info(f"Image topic       : {self.image_topic}")
        self.get_logger().info(f"CameraInfo topic  : {self.camera_info_topic}")
        self.get_logger().info(f"Parent frame      : {self.parent_frame}")
        self.get_logger().info(f"ChArUco frame     : {self.charuco_frame}")
        self.get_logger().info(
            f"Board             : {self.squares_x} x {self.squares_y}, "
            f"square={self.square_length} m, marker={self.marker_length} m"
        )

    def get_dictionary(self, name):
        dictionaries = {
            "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
            "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
            "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
            "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
            "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
            "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
            "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
            "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
            "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
            "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
            "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
            "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
            "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
            "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
            "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
            "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
        }

        if name not in dictionaries:
            self.get_logger().warn(
                f"Unknown dictionary {name}, fallback to DICT_6X6_250"
            )
            name = "DICT_6X6_250"

        return cv2.aruco.getPredefinedDictionary(dictionaries[name])

    def create_charuco_board(self):
        """
        OpenCV 不同版本的 ChArUco API 有差異，所以這裡做相容處理。
        """
        try:
            # OpenCV 較新版
            board = cv2.aruco.CharucoBoard(
                (self.squares_x, self.squares_y),
                self.square_length,
                self.marker_length,
                self.dictionary
            )
        except Exception:
            # OpenCV 較舊版
            board = cv2.aruco.CharucoBoard_create(
                self.squares_x,
                self.squares_y,
                self.square_length,
                self.marker_length,
                self.dictionary
            )

        return board

    def camera_info_callback(self, msg: CameraInfo):
        k = np.array(msg.k, dtype=np.float64).reshape((3, 3))

        if np.allclose(k, 0.0):
            self.get_logger().warn("Received CameraInfo, but K is all zero")
            return

        self.camera_matrix = k
        self.dist_coeffs = np.array(msg.d, dtype=np.float64).reshape((1, -1))

        self.camera_info_received = True

        if not hasattr(self, "printed_camera_info"):
            self.printed_camera_info = False

        if not self.printed_camera_info:
            self.get_logger().info(
                "CameraInfo received: "
                f"frame_id={msg.header.frame_id}, "
                f"fx={self.camera_matrix[0, 0]:.3f}, "
                f"fy={self.camera_matrix[1, 1]:.3f}, "
                f"cx={self.camera_matrix[0, 2]:.3f}, "
                f"cy={self.camera_matrix[1, 2]:.3f}, "
                f"dist_size={len(msg.d)}"
            )
            self.printed_camera_info = True

    def image_callback(self, msg: Image):
        self.get_logger().info(f"Image received: encoding={msg.encoding}, size={msg.width}x{msg.height}")

        if not self.camera_info_received:
            self.get_logger().warn("Waiting for CameraInfo...")
            return

        try:
            if msg.encoding == "mono8":
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            else:
                img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

                if msg.encoding == "bgra8":
                    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                elif msg.encoding == "rgba8":
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                elif msg.encoding == "bgr8":
                    bgr = img

                elif msg.encoding == "rgb8":
                    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                else:
                    self.get_logger().warn(f"Unsupported image encoding: {msg.encoding}")
                    return

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        except Exception as e:
            self.get_logger().warn(f"cv_bridge failed: {e}")
            return

        if self.use_clahe:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
            gray = clahe.apply(gray)

        debug_image = bgr.copy()

        if hasattr(cv2.aruco, "detectMarkers"):
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray,
                self.dictionary,
                parameters=self.detector_params
            )
        else:
            aruco_detector = cv2.aruco.ArucoDetector(
                self.dictionary,
                self.detector_params
            )
            corners, ids, rejected = aruco_detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            self.publish_debug_image(msg, debug_image)
            self.get_logger().warn("No ArUco markers detected")
            return

        # refineDetectedMarkers 有些版本對 Python binding 比較挑，所以用 try 避免直接炸掉
        try:
            cv2.aruco.refineDetectedMarkers(
                image=gray,
                board=self.board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=rejected,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs
            )
        except Exception:
            pass

        charuco_corners, charuco_ids = self.interpolate_charuco(
            corners,
            ids,
            gray
        )

        cv2.aruco.drawDetectedMarkers(debug_image, corners, ids)

        if charuco_ids is None or len(charuco_ids) < 4:
            self.publish_debug_image(msg, debug_image)
            self.get_logger().warn(
                f"Not enough ChArUco corners. markers={len(ids)}, "
                f"charuco_corners={0 if charuco_ids is None else len(charuco_ids)}"
            )
            return

        cv2.aruco.drawDetectedCornersCharuco(
            debug_image,
            charuco_corners,
            charuco_ids,
            (255, 0, 0)
        )

        valid, rvec, tvec = self.estimate_charuco_pose(
            charuco_corners,
            charuco_ids
        )

        if not valid:
            self.publish_debug_image(msg, debug_image)
            self.get_logger().warn("Pose estimation failed")
            return

        # 畫出座標軸
        axis_length = min(self.squares_x, self.squares_y) * self.square_length * 0.5
        try:
            cv2.drawFrameAxes(
                debug_image,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                axis_length
            )
        except Exception:
            pass

        qx, qy, qz, qw = self.rvec_to_quaternion(rvec)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = msg.header.stamp

        if self.parent_frame:
            pose_msg.header.frame_id = self.parent_frame
        else:
            pose_msg.header.frame_id = msg.header.frame_id

        pose_msg.pose.position.x = float(tvec[0][0])
        pose_msg.pose.position.y = float(tvec[1][0])
        pose_msg.pose.position.z = float(tvec[2][0])

        pose_msg.pose.orientation.x = float(qx)
        pose_msg.pose.orientation.y = float(qy)
        pose_msg.pose.orientation.z = float(qz)
        pose_msg.pose.orientation.w = float(qw)

        self.pose_pub.publish(pose_msg)

        if self.publish_tf:
            tf_msg = TransformStamped()
            tf_msg.header = pose_msg.header
            tf_msg.child_frame_id = self.charuco_frame

            tf_msg.transform.translation.x = pose_msg.pose.position.x
            tf_msg.transform.translation.y = pose_msg.pose.position.y
            tf_msg.transform.translation.z = pose_msg.pose.position.z

            tf_msg.transform.rotation = pose_msg.pose.orientation

            self.tf_broadcaster.sendTransform(tf_msg)

        self.get_logger().info(
            "Camera -> ChArUco [m] "
            f"x={pose_msg.pose.position.x:.4f}, "
            f"y={pose_msg.pose.position.y:.4f}, "
            f"z={pose_msg.pose.position.z:.4f} | "
            f"qx={qx:.4f}, qy={qy:.4f}, qz={qz:.4f}, qw={qw:.4f} | "
            f"parent={pose_msg.header.frame_id}, child={self.charuco_frame}, "
            f"corners={len(charuco_ids)}"
        )

        self.publish_debug_image(msg, debug_image)

    def interpolate_charuco(self, corners, ids, gray):
        try:
            # 常見 OpenCV 4.x 寫法
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                markerCorners=corners,
                markerIds=ids,
                image=gray,
                board=self.board,
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs
            )
            return charuco_corners, charuco_ids
        except Exception:
            # 另一種 binding 寫法
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners,
                ids,
                gray,
                self.board,
                self.camera_matrix,
                self.dist_coeffs
            )
            return charuco_corners, charuco_ids

    def estimate_charuco_pose(self, charuco_corners, charuco_ids):
        rvec = np.zeros((3, 1), dtype=np.float64)
        tvec = np.zeros((3, 1), dtype=np.float64)

        try:
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec
            )
            return bool(valid), rvec, tvec
        except Exception as e:
            self.get_logger().warn(f"estimatePoseCharucoBoard failed: {e}")
            return False, rvec, tvec

    def rvec_to_quaternion(self, rvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        m00 = rotation_matrix[0, 0]
        m01 = rotation_matrix[0, 1]
        m02 = rotation_matrix[0, 2]

        m10 = rotation_matrix[1, 0]
        m11 = rotation_matrix[1, 1]
        m12 = rotation_matrix[1, 2]

        m20 = rotation_matrix[2, 0]
        m21 = rotation_matrix[2, 1]
        m22 = rotation_matrix[2, 2]

        trace = m00 + m11 + m22

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m21 - m12) / s
            qy = (m02 - m20) / s
            qz = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            qw = (m21 - m12) / s
            qx = 0.25 * s
            qy = (m01 + m10) / s
            qz = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            qw = (m02 - m20) / s
            qx = (m01 + m10) / s
            qy = 0.25 * s
            qz = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            qw = (m10 - m01) / s
            qx = (m02 + m20) / s
            qy = (m12 + m21) / s
            qz = 0.25 * s

        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

        if norm > 1e-12:
            qx /= norm
            qy /= norm
            qz /= norm
            qw /= norm

        return qx, qy, qz, qw

    def publish_debug_image(self, src_msg, debug_image):
        if not self.show_debug_image:
            return

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            debug_msg.header = src_msg.header
            self.debug_image_pub.publish(debug_msg)
            self.get_logger().info("Published /charuco/debug_image")
        except Exception as e:
            self.get_logger().warn(f"Publish debug image failed: {e}")


def main(args=None):
    rclpy.init(args=args)

    node = AzureCharucoPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()