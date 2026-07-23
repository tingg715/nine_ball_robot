import cv2
import numpy as np
import os
import time

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from detect_interface.msg import DetectionResults, BoundingBox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ament_index_python.packages import get_package_share_directory

from ultralytics import YOLO


class ObjectDetection(Node):
    def __init__(self):
        super().__init__("object_detection")

        # Parameters
        self.declare_parameter(
            "weights",
            "best.pt",
            ParameterDescriptor(description="Weights file")
        )
        self.declare_parameter(
            "conf_thres",
            0.25,
            ParameterDescriptor(description="Confidence threshold")
        )
        self.declare_parameter(
            "iou_thres",
            0.45,
            ParameterDescriptor(description="IOU threshold")
        )
        self.declare_parameter(
            "device",
            "",
            ParameterDescriptor(description="Device, for example: '', 'cpu', '0'")
        )
        self.declare_parameter(
            "img_size",
            640,
            ParameterDescriptor(description="Image size")
        )
        self.declare_parameter(
            "show_img",
            True,
            ParameterDescriptor(description="Show detection image")
        )

        weights_file = self.get_parameter("weights").get_parameter_value().string_value
        self.weights = os.path.join(
            get_package_share_directory("yolov7_obj_detect"),
            "weights",
            weights_file
        )

        self.conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        self.iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.img_size = self.get_parameter("img_size").get_parameter_value().integer_value
        self.show_img = self.get_parameter("show_img").get_parameter_value().bool_value

        self.cv_bridge = CvBridge()
        self.rgb_image = None

        self.get_logger().info(f"Loading YOLO model: {self.weights}")

        # Load YOLOv12 / Ultralytics model
        self.model = YOLO(self.weights)

        # Class names
        self.names = self.model.names

        self.get_logger().info(f"Model classes: {self.names}")

        # Random colors for drawing boxes
        self.colors = [
            [int(np.random.randint(0, 255)) for _ in range(3)]
            for _ in range(len(self.names))
        ]

        # Subscriber: Azure Kinect RGB image
        self.rs_sub = self.create_subscription(
            Image,
            "/camera_calib",
            self.rs_callback,
            1
        )

        # Publisher: detection results
        self.publisher_obj = self.create_publisher(
            DetectionResults,
            "/detect/objs",
            1
        )

        self.get_logger().info("ObjectDetection node started.")
        self.get_logger().info("Subscribing to /camera_calib")
        self.get_logger().info("Publishing to /detect/objs")

    def rs_callback(self, msg):
        try:
            self.rgb_image = self.cv_bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        self.yolo_detect()

    def yolo_detect(self):
        if self.rgb_image is None:
            return

        im0 = self.rgb_image.copy()

        t1 = time.time()

        try:
            results = self.model(
                im0,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.img_size,
                device=self.device if self.device != "" else None,
                verbose=False
            )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        t2 = time.time()

        detections = DetectionResults()

        # Ultralytics results normally contains one result because input is one image
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                try:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy

                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())

                    if cls_id not in self.names:
                        self.get_logger().warn(
                            f"Unknown class id: {cls_id}, names: {self.names}"
                        )
                        continue

                    cls_name = str(self.names[cls_id])

                    bbox = BoundingBox(
                        xmin=int(x1),
                        ymin=int(y1),
                        xmax=int(x2),
                        ymax=int(y2)
                    )

                    detections.labels.append(cls_name)
                    detections.scores.append(conf)
                    detections.bounding_boxes.append(bbox)

                    if self.show_img:
                        color = self.colors[cls_id % len(self.colors)]
                        label = f"{cls_name} {conf:.2f}"

                        cv2.rectangle(
                            im0,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            color,
                            2
                        )

                        cv2.putText(
                            im0,
                            label,
                            (int(x1), max(int(y1) - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                except Exception as e:
                    self.get_logger().warn(f"Failed to parse detection box: {e}")
                    continue

        self.publisher_obj.publish(detections)

        self.get_logger().info(
            f"Published {len(detections.labels)} detections | "
            f"Inference time: {t2 - t1:.3f} sec"
        )

        if self.show_img:
            # 取得原始影像尺寸
            image_height, image_width = im0.shape[:2]

            # 計算畫面中心
            center_x = image_width // 2
            center_y = image_height // 2

            # 畫中心點
            cv2.circle(
                im0,
                (center_x, center_y),
                8,              # 圓點半徑
                (0, 0, 255),    # BGR：紅色
                -1              # 填滿
            )

            # 畫十字線，方便觀察
            cv2.line(
                im0,
                (center_x - 25, center_y),
                (center_x + 25, center_y),
                (0, 255, 255),  # 黃色
                2
            )

            cv2.line(
                im0,
                (center_x, center_y - 25),
                (center_x, center_y + 25),
                (0, 255, 255),
                2
            )

            # 顯示中心座標
            cv2.putText(
                im0,
                f"Center ({center_x}, {center_y})",
                (center_x + 12, center_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            show = cv2.resize(im0, None, fx=0.5, fy=0.5)

            cv2.imshow("YOLOv12 Object Detection RGB", show)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)

    node = ObjectDetection()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()