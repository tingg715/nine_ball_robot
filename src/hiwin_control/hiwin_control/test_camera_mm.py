#!/usr/bin/env python3
import os
import math
import yaml
import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


# RGB camera intrinsic parameters from /rgb/camera_info
IMG_W = 1920
IMG_H = 1080

FX = 903.7640991210938
FY = 903.706787109375
CX = 961.153564453125
CY = 546.2996826171875


def pixel_mm_convert(cam_to_table_h, pixels):
    u = pixels[0]
    v = pixels[1]

    dev_x = (u - CX) * cam_to_table_h / FX
    dev_y = (v - CY) * cam_to_table_h / FY

    return [dev_x, dev_y]


class TestCameraMM(Node):
    def __init__(self):
        super().__init__('test_camera_mm')

        current_dir = os.getcwd()
        yaml_path = current_dir + '/src/hiwin_control/hiwin_control/arm.yaml'

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        self.cam_to_table = data['zoff']

        self.coords = []
        self.labels = []

        self.create_subscription(
            Float64MultiArray,
            'center_data_coords',
            self.coord_callback,
            10
        )

        self.create_subscription(
            String,
            'center_data_labels',
            self.label_callback,
            10
        )

        self.get_logger().info('test_camera_mm started')
        self.get_logger().info(f'CAM_TO_TABLE / zoff = {self.cam_to_table} mm')
        self.get_logger().info(f'FX={FX}, FY={FY}, CX={CX}, CY={CY}')

    def label_callback(self, msg):
        try:
            self.labels = json.loads(msg.data.replace("'", '"'))
        except Exception:
            try:
                self.labels = eval(msg.data)
            except Exception:
                self.labels = []

    def coord_callback(self, msg):
        self.coords = list(msg.data)

        print('\n========== CAMERA MM TEST ==========')
        print(f'zoff = {self.cam_to_table} mm')
        print(f'optical center = ({CX:.2f}, {CY:.2f})')

        for i in range(0, len(self.coords), 2):
            u = self.coords[i]
            v = self.coords[i + 1]

            label = self.labels[i // 2] if i // 2 < len(self.labels) else str(i // 2)

            cam_x, cam_y = pixel_mm_convert(self.cam_to_table, [u, v])

            pixel_dx = u - CX
            pixel_dy = v - CY

            print(
                f'ball {label:>5} | '
                f'pixel=({u:8.2f}, {v:8.2f}) | '
                f'pixel_offset=({pixel_dx:8.2f}, {pixel_dy:8.2f}) | '
                f'camera_mm=({cam_x:8.2f}, {cam_y:8.2f})'
            )


def main(args=None):
    rclpy.init(args=args)
    node = TestCameraMM()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()