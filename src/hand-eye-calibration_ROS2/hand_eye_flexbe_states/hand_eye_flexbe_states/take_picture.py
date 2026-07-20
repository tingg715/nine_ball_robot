#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
take_picture.py

FlexBE State:
- camera_type == "realsense": use pyrealsense2 directly
- camera_type == "azure_kinect": subscribe ROS2 Image topic, e.g. /rgb/image_raw
- press Enter to save images
- press q or Esc to fail/quit

Default save path:
<charuco_detector share>/config/camera_calibration/pic/
"""

from flexbe_core import EventState, Logger

import cv2
import numpy as np
import os

from ament_index_python.packages import get_package_share_directory

# ROS2 image topic mode for Azure Kinect
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# RealSense mode is optional
try:
    import pyrealsense2 as rs
except Exception:
    rs = None


class TakePictureState(EventState):
    """
    Take ChArUco calibration pictures.

    <= done      Saved enough pictures.
    <= failed    Failed or user quit.

    camera_type:
      - "realsense"
      - "azure_kinect"

    For Azure Kinect, this state subscribes to:
      /rgb/image_raw

    Press Enter to save one image.
    Press q or Esc to quit with failed.
    """

    def __init__(self, pic_num, camera_type):
        super(TakePictureState, self).__init__(outcomes=['done', 'failed'])

        self.excu_num = 1
        self.pic_num = int(pic_num)
        self.camera_type = str(camera_type)

        self.bridge = CvBridge()
        self.latest_image = None
        self.latest_encoding = None
        self.image_sub = None

        self.pipeline = None
        self.config = None
        self.cfg = None

        # Default Azure Kinect topic from azure_kinect_ros_driver
        self.image_topic = "/rgb/image_raw"

        # Save folder used by CharucoCameraCalibrationState
        self.save_pwd = (
            get_package_share_directory('charuco_detector')
            + '/config/camera_calibration/pic/'
        )
        os.makedirs(self.save_pwd, exist_ok=True)

        Logger.logwarn("TakePicture save path: " + self.save_pwd)
        Logger.logwarn("TakePicture camera_type: " + self.camera_type)

        if self.camera_type == 'realsense':
            if rs is None:
                Logger.logwarn("pyrealsense2 is not installed, cannot use realsense mode.")
                return

            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
            self.cfg = self.pipeline.start(self.config)
            Logger.logwarn("RealSense pipeline started.")

        elif self.camera_type == 'azure_kinect':
            # Subscribe in __init__ so images are already buffered before execute().
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )

            self.image_sub = TakePictureState._node.create_subscription(
                Image,
                self.image_topic,
                self._image_callback,
                qos
            )
            Logger.logwarn("Azure Kinect image topic: " + self.image_topic)

        else:
            Logger.logwarn("Unsupported camera_type: " + self.camera_type)

    def _image_callback(self, msg: Image):
        """
        Store the newest image from ROS topic.
        Azure Kinect RGB image is commonly bgra8.
        """
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            if msg.encoding == "bgra8":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif msg.encoding == "rgba8":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif msg.encoding == "rgb8":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgr8":
                img_bgr = img
            elif msg.encoding == "mono8":
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                Logger.logwarn("Unsupported Azure Kinect image encoding: " + msg.encoding)
                return

            self.latest_image = img_bgr
            self.latest_encoding = msg.encoding

        except Exception as e:
            Logger.logwarn("Azure Kinect image conversion failed: " + str(e))

    def on_start(self):
        pass

    def on_enter(self, userdata):
        self.enter_time = TakePictureState._node.get_clock().now()
        print(self.enter_time)

    def execute(self, userdata):
        if self.camera_type not in ['realsense', 'azure_kinect']:
            Logger.logwarn("Unsupported camera_type: " + self.camera_type)
            return "failed"

        while True:
            images = None

            if self.camera_type == 'realsense':
                if self.pipeline is None:
                    Logger.logwarn("RealSense pipeline is not available.")
                    return "failed"

                try:
                    frames = self.pipeline.wait_for_frames()
                    color_frame = frames.get_color_frame()

                    if not color_frame:
                        Logger.logwarn("Failed to get RealSense color frame.")
                        continue

                    images = np.asanyarray(color_frame.get_data())

                except Exception as e:
                    Logger.logwarn("RealSense capture failed: " + str(e))
                    return "failed"

            elif self.camera_type == 'azure_kinect':
                if self.latest_image is None:
                    Logger.logwarn("Waiting for Azure Kinect image from " + self.image_topic)
                    # Keep GUI responsive even before first image.
                    key = cv2.waitKey(30)
                    if key & 0xFF == ord('q') or key == 27:
                        self._cleanup()
                        return "failed"
                    continue

                images = self.latest_image.copy()

            cv2.namedWindow('preview', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('preview', images)

            key = cv2.waitKey(1)

            # Enter
            if key == 13:
                Logger.logwarn("----------------------------------------------")
                filename = (
                    self.save_pwd
                    + "camera-pic-of-charucoboard-"
                    + str(self.excu_num)
                    + ".jpg"
                )

                ok = cv2.imwrite(filename, images)

                if ok:
                    Logger.logwarn("Saved: " + filename)
                    self.excu_num += 1
                else:
                    Logger.logwarn("Failed to save: " + filename)
                    return "failed"

            # q or Esc
            elif key & 0xFF == ord('q') or key == 27:
                Logger.logwarn("==========================================")
                self.excu_num = self.pic_num
                self._cleanup()
                return "failed"

            # Done
            elif self.excu_num >= self.pic_num + 1:
                self._cleanup()
                return "done"

    def _cleanup(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
                Logger.logwarn("RealSense pipeline stopped.")
            except Exception:
                pass