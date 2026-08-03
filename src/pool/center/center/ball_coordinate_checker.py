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
# Y 軸二次補償
#
# 先將像素座標正規化：
#
# U = (pixel_u - 960) / 960
# V = (pixel_v - 540) / 540
#
# Y 補償：
#
# delta_y =
#     C0
#     + CU  * U
#     + CV  * V
#     + CUU * U^2
#     + CUV * U*V
#     + CVV * V^2
#
# 最終：
#
# final_x = raw_homography_x
# final_y = raw_homography_y + delta_y
# ============================================================

PIXEL_CENTER_U = 960.0
PIXEL_CENTER_V = 540.0

PIXEL_SCALE_U = 960.0
PIXEL_SCALE_V = 540.0


Y_QUADRATIC_C0 = 2.1609174167
Y_QUADRATIC_U = -1.0376556200
Y_QUADRATIC_V = 1.8120073500
Y_QUADRATIC_UU = 0.8814067500
Y_QUADRATIC_UV = 2.7975178900
Y_QUADRATIC_VV = -2.4531395200


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

        # 發布所有球的最終 Base X、Y 座標
        self.ball_coordinate_publisher = self.create_publisher(
            String,
            'ball_coordinate',
            10,
        )

        self.get_logger().info(
            'Ball coordinate checker started.'
        )

        self.get_logger().info(
            'Subscribe: /center_data_labels'
        )

        self.get_logger().info(
            'Subscribe: /center_data_coords'
        )

        self.get_logger().info(
            'Publish: /ball_coordinate'
        )

        self.get_logger().info(
            'X correction: disabled.'
        )

        self.get_logger().info(
            'Single quadratic Y correction: enabled.'
        )

    # ========================================================
    # 接收球標籤
    #
    # 預期格式：
    # "['1', '2', '8', 'white']"
    # ========================================================
    def label_callback(
        self,
        msg: String,
    ) -> None:

        try:
            parsed_labels = ast.literal_eval(
                msg.data
            )

            if not isinstance(
                parsed_labels,
                (list, tuple),
            ):
                self.get_logger().warning(
                    'center_data_labels is not a list or tuple.'
                )
                return

            self.labels = [
                str(label)
                for label in parsed_labels
            ]

        except (
            ValueError,
            SyntaxError,
        ) as error:

            self.get_logger().warning(
                f'Unable to parse center_data_labels: {error}'
            )

    # ========================================================
    # 接收像素座標並轉換成 Base 座標
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

        # 每一顆球必須包含 u、v 兩個數值
        if len(coordinates) % 2 != 0:
            self.get_logger().error(
                'Coordinate number must be even. '
                f'Received {len(coordinates)} values.'
            )
            return

        # 將資料整理成：
        #
        # [
        #   [u1, v1],
        #   [u2, v2],
        #   ...
        # ]
        ball_pixels = np.asarray(
            coordinates,
            dtype=np.float64,
        ).reshape(-1, 2)

        ball_count = len(ball_pixels)

        # 標籤數量與球座標數量必須一致
        if len(self.labels) != ball_count:
            self.get_logger().warning(
                'Labels and coordinates count do not match: '
                f'labels={len(self.labels)}, '
                f'coordinates={ball_count}.'
            )
            return

        converted_balls = []

        self.get_logger().info(
            '\n'
            '========================================\n'
            f'Detected balls: {ball_count}\n'
            '========================================'
        )

        for index, pixel in enumerate(ball_pixels):

            label = self.labels[index]

            pixel_u = float(pixel[0])
            pixel_v = float(pixel[1])

            try:
                (
                    final_base_x,
                    final_base_y,
                    raw_base_x,
                    raw_base_y,
                    y_correction,
                    normalized_u,
                    normalized_v,
                ) = self.pixel_to_base(
                    pixel_u,
                    pixel_v,
                )

            except ValueError as error:
                self.get_logger().error(
                    f'Ball {label} conversion failed: {error}'
                )
                continue

            # 發布的資料只保留 label、x、y
            #
            # 這裡的 x：
            # 原始 Homography X
            #
            # 這裡的 y：
            # 原始 Homography Y + 二次 Y 補償
            ball_data = {
                'label': label,
                'x': round(final_base_x, 3),
                'y': round(final_base_y, 3),
            }

            converted_balls.append(
                ball_data
            )

            # 終端顯示每顆球的完整轉換資訊
            self.get_logger().info(
                '\n'
                f'Ball {index + 1}: {label}\n'
                f'  pixel (u, v) = '
                f'({pixel_u:.2f}, {pixel_v:.2f})\n'
                f'  normalized (U, V) = '
                f'({normalized_u:.6f}, '
                f'{normalized_v:.6f})\n'
                f'  raw Homography (x, y) mm = '
                f'({raw_base_x:.3f}, '
                f'{raw_base_y:.3f})\n'
                f'  Y correction mm = '
                f'{y_correction:+.3f}\n'
                f'  final base (x, y) mm = '
                f'({final_base_x:.3f}, '
                f'{final_base_y:.3f})'
            )

        # 沒有成功轉換任何球時，不發布
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
    # Pixel -> Base Homography + 單一二次 Y 補償
    # ========================================================
    @staticmethod
    def pixel_to_base(
        pixel_u: float,
        pixel_v: float,
    ) -> Tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:

        pixel_point = np.array(
            [
                pixel_u,
                pixel_v,
                1.0,
            ],
            dtype=np.float64,
        )

        transformed_point = (
            PIXEL_TO_BASE_H
            @ pixel_point
        )

        denominator = float(
            transformed_point[2]
        )

        if abs(denominator) < 1e-10:
            raise ValueError(
                'Homography denominator is too close to zero.'
            )

        # ----------------------------------------------------
        # 原始 Homography X
        # ----------------------------------------------------
        raw_base_x = (
            float(transformed_point[0])
            / denominator
        )

        # ----------------------------------------------------
        # 原始 Homography Y
        # ----------------------------------------------------
        raw_base_y = (
            float(transformed_point[1])
            / denominator
        )

        # ----------------------------------------------------
        # 正規化像素座標
        # ----------------------------------------------------
        normalized_u = (
            pixel_u
            - PIXEL_CENTER_U
        ) / PIXEL_SCALE_U

        normalized_v = (
            pixel_v
            - PIXEL_CENTER_V
        ) / PIXEL_SCALE_V

        # ----------------------------------------------------
        # 單一二次 Y 補償
        # ----------------------------------------------------
        y_correction = (
            Y_QUADRATIC_C0
            + Y_QUADRATIC_U
            * normalized_u
            + Y_QUADRATIC_V
            * normalized_v
            + Y_QUADRATIC_UU
            * normalized_u
            * normalized_u
            + Y_QUADRATIC_UV
            * normalized_u
            * normalized_v
            + Y_QUADRATIC_VV
            * normalized_v
            * normalized_v
        )

        # ----------------------------------------------------
        # X 不做額外補償
        # ----------------------------------------------------
        final_base_x = raw_base_x

        # ----------------------------------------------------
        # Y 只加入這一套二次補償
        # ----------------------------------------------------
        final_base_y = (
            raw_base_y
            + y_correction
        )

        return (
            final_base_x,
            final_base_y,
            raw_base_x,
            raw_base_y,
            y_correction,
            normalized_u,
            normalized_v,
        )


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

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()