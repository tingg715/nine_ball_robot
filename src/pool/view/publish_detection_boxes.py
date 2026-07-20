import cv2
import rclpy
import numpy as np
import pyrealsense2 as rs
from rclpy.node import Node
from detect_interface.msg import DetectionResults, BoundingBox
from std_msgs.msg import String  # 使用標準消息類型，或根據需要自定義

# 初始化攝影機
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1920, 1080, rs.format.z16, 30) #深度數據流設定
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
pipeline.start(config)

class BoxPublisher(Node):
    def __init__(self):
        super().__init__('box_publisher')
        self.subscription = self.create_subscription(DetectionResults,'/detect/objs',self.detection_callback,10)
        self.publisher = self.create_publisher(String, 'center_data', 10)  # 增加一個發布者
        self.get_logger().info('BoxPublisher has been started and is subscribing and publishing.')

    def detection_callback(self, msg):
        # 從攝影機讀取畫面
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return
        #記錄一條信息，表收到檢測結果。
        self.get_logger().info('Received detection results')
        #轉換攝影機畫面為numpy array
        color_image = np.asanyarray(color_frame.get_data())
        #使用for循環:每個檢測到的物體。
        for label, score, bbox in zip(msg.labels, msg.scores, msg.bounding_boxes):
            # 計算 BoundingBox 的中心點
            center_x = (bbox.xmin + bbox.xmax) / 2
            center_y = (bbox.ymin + bbox.ymax) / 2
            # 處理數據:將每個物體的標籤、得分和邊界框信息格式化成一個字符串 
            #processed_data = f'Detected {label} with score {score:.2f} at [{bbox.xmin}, {bbox.ymin}, {bbox.xmax}, {bbox.ymax}, center: ({center_x:.2f}, {center_y:.2f})'
            #僅發布中心點
            center_data = f'Center of {label}: ({center_x:.2f}, {center_y:.2f})'
            self.publisher.publish(String(data=center_data))  # 發布處理後的數據
            self.get_logger().info(f'Published: {center_data}')
         # 顯示畫面
        cv2.imshow('YOLOv7 Realtime Detection', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             return

def main(args=None):
    rclpy.init(args=args)
    node = BoxPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
