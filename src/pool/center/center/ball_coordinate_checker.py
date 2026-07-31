
#!/usr/bin/env python3

import ast
import json
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String


# ============================================================
# Pixel (u, v) -> Robot Base (x, y) Homography
# ============================================================
PIXEL_TO_BASE_H = np.array([
    [
        0.6953817196638,
        -0.0049199588166,
        -663.7505877708,
    ],
    [
        -0.0001255227889,
        -0.7173316193511,
        835.0951639315,
    ],
    [
        0.0000080206744,
        -0.0000012882134,
        1.0000000000,
    ],
], dtype=np.float64)


# ============================================================
# Y 軸殘差補償
#
# delta_y =
#     C0
#     + CU * pixel_u
#     + CV * pixel_v
# ============================================================
Y_CORRECTION_C0 = 4.17110130
Y_CORRECTION_U = -0.00271929316
Y_CORRECTION_V = 0.00578073253


class BallCoordinateChecker(Node):

    def __init__(self) -> None:
        super().__init__('ball_coordinate_checker')

        # 最新收到的球標籤
        self.labels: List[str] = []

        # 接收球標籤
        self.create_subscription(
            String,
            'center_data_labels',
            self.label_callback,
            10,
        )

        # 接收球中心像素座標
        self.create_subscription(
            Float64MultiArray,
            'center_data_coords',
            self.coordinate_callback,
            10,
        )

        # 發布所有球的 Base X、Y 座標
        self.ball_coordinate_publisher = self.create_publisher(
            String,
            'ball_coordinate',
            10,
        )

        self.get_logger().info(
            'Ball coordinate publisher started.'
        )

        self.get_logger().info(
            'Publishing Base coordinates to /ball_coordinate'
        )

    # ========================================================
    # 接收標籤
    #
    # 預期格式：
    # "['1', '2', '8', 'white']"
    # ========================================================
    def label_callback(self, msg: String) -> None:
        try:
            parsed_labels = ast.literal_eval(msg.data)

            if not isinstance(parsed_labels, (list, tuple)):
                self.get_logger().warning(
                    'center_data_labels is not a list or tuple.'
                )
                return

            self.labels = [
                str(label)
                for label in parsed_labels
            ]

        except (ValueError, SyntaxError) as error:
            self.get_logger().warning(
                f'Unable to parse center_data_labels: {error}'
            )

    # ========================================================
    # 接收像素座標並轉換
    #
    # 預期格式：
    # [u1, v1, u2, v2, ...]
    # ========================================================
    def coordinate_callback(
        self,
        msg: Float64MultiArray,
    ) -> None:

        coordinates = list(msg.data)

        if not coordinates:
            return

        if len(coordinates) % 2 != 0:
            self.get_logger().error(
                'Coordinate number must be even. '
                f'Received {len(coordinates)} values.'
            )
            return

        ball_pixels = np.asarray(
            coordinates,
            dtype=np.float64,
        ).reshape(-1, 2)

        ball_count = len(ball_pixels)

        # 標籤和座標數量必須相同
        if len(self.labels) != ball_count:
            self.get_logger().warning(
                'Labels and coordinates count do not match: '
                f'labels={len(self.labels)}, '
                f'coordinates={ball_count}.'
            )
            return

        converted_balls = []

        for index, pixel in enumerate(ball_pixels):

            label = self.labels[index]

            pixel_u = float(pixel[0])
            pixel_v = float(pixel[1])

            try:
                base_x, base_y = self.pixel_to_base(
                    pixel_u,
                    pixel_v,
                )

            except ValueError as error:
                self.get_logger().error(
                    f'Ball {label} conversion failed: {error}'
                )
                continue

            # 每顆球只發布 label、x、y
            converted_balls.append({
                'label': label,
                'x': round(base_x, 3),
                'y': round(base_y, 3),
            })

        # 如果沒有成功轉換任何球，就不發布
        if not converted_balls:
            return

        output_message = String()

        output_message.data = json.dumps(
            converted_balls,
            ensure_ascii=False,
        )

        self.ball_coordinate_publisher.publish(
            output_message
        )

    # ========================================================
    # Pixel -> Base Homography + Y 補償
    # ========================================================
    @staticmethod
    def pixel_to_base(
        pixel_u: float,
        pixel_v: float,
    ) -> Tuple[float, float]:

        pixel_point = np.array(
            [
                pixel_u,
                pixel_v,
                1.0,
            ],
            dtype=np.float64,
        )

        transformed_point = (
            PIXEL_TO_BASE_H @ pixel_point
        )

        denominator = float(
            transformed_point[2]
        )

        if abs(denominator) < 1e-10:
            raise ValueError(
                'Homography denominator is too close to zero.'
            )

        base_x = (
            float(transformed_point[0])
            / denominator
        )

        raw_base_y = (
            float(transformed_point[1])
            / denominator
        )

        # Y 軸殘差補償
        y_correction = (
            Y_CORRECTION_C0
            + Y_CORRECTION_U * pixel_u
            + Y_CORRECTION_V * pixel_v
        )

        base_y = raw_base_y + y_correction

        return base_x, base_y


def main(
    args: Optional[List[str]] = None,
) -> None:

    rclpy.init(args=args)

    node: Optional[BallCoordinateChecker] = None

    try:
        node = BallCoordinateChecker()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

