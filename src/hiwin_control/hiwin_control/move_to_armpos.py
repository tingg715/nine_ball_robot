#!/usr/bin/env python3
"""
One-shot helper: move the arm to arm.yaml -> armpos (the photo pose) and stop.

Purpose: put the flange exactly where ball_coordinate_checker.py assumes the
camera is, so you can safely test first-photo accuracy WITHOUT running the
DYNAMIC_CALI servo loop.

IMPORTANT:
  - armpos was recorded with tool=0, base=0 (the flange), so we move with
    tool=0, base=0 to land exactly there.
  - Make sure the arm starts from a safe position and the area is clear before
    running: this sends a single PTP straight to armpos.

Run (from the workspace root, after sourcing install/setup.bash):
  python3 src/hiwin_control/hiwin_control/move_to_armpos.py
"""
import os
import time

import yaml
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Twist
from hiwin_interfaces.srv import RobotCommand

VELOCITY = 30       # slow for safety; raise if you want it faster
ACCELERATION = 30


def load_armpos():
    path = os.path.join(os.getcwd(),
                        'src/hiwin_control/hiwin_control/arm.yaml')
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return data['armpos']


class MoveToArmpos(Node):
    def __init__(self):
        super().__init__('move_to_armpos')
        self.client = self.create_client(RobotCommand, 'hiwinmodbus_service')

    def call(self, req):
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('waiting for hiwinmodbus_service...')
        future = self.client.call_async(req)
        self._wait(future)
        return future.result()

    def _wait(self, future: Future):
        while not future.done():
            time.sleep(0.01)

    def move(self, armpos):
        pose = Twist()
        [pose.linear.x, pose.linear.y, pose.linear.z] = armpos[0:3]
        [pose.angular.x, pose.angular.y, pose.angular.z] = armpos[3:6]

        req = RobotCommand.Request()
        req.cmd_mode = RobotCommand.Request.PTP
        req.cmd_type = RobotCommand.Request.POSE_CMD
        req.pose = pose
        req.tool = 0
        req.base = 0
        req.velocity = VELOCITY
        req.acceleration = ACCELERATION
        req.holding = True

        self.get_logger().info(f'Moving to armpos: {armpos}')
        res = self.call(req)
        if res is None:
            self.get_logger().error('No response from robot service.')
            return

        # Read back where the arm actually ended up.
        check = RobotCommand.Request()
        check.cmd_mode = RobotCommand.Request.CHECK_POSE
        check.tool = 0
        check.base = 0
        res = self.call(check)
        self.get_logger().info(f'Arm actual position now: {list(res.current_position)}')
        self.get_logger().info('Done. Arm is at armpos and stopped.')


def main(args=None):
    rclpy.init(args=args)
    node = MoveToArmpos()
    try:
        node.move(load_armpos())
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
