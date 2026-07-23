#!/usr/bin/env python3
"""
One-shot helper: move the cue's hitting end (tool 8) to directly ABOVE the ball
nearest the image center, then stop. It does NOT descend and does NOT hit.

Purpose: verify the whole pixel -> mm -> base-frame coordinate chain by parking
the hitting-end TCP right over a real ball, so you can eye-check the XY accuracy
before trusting it for a real shot.

It reuses the exact production math from arm_controler_how.py
(pixel_mm_convert, convert_arm_pose, check_mid_pose, and every calibration
constant) instead of copying it, so this test always tracks whatever the real
program uses -- no duplicated / drifting calibration.

Flow:
  1. flange -> FIX_ABS_CAM (armpos) photo pose                    [tool 0]
  2. wait for YOLO, pick the ball nearest the image center
  3. pixel -> mm -> base-frame XY  (one-shot, first photo only)
  4. tool 8 -> [x, y, ABOVE_Z], keeping the current orientation   [tool 8]
  5. read the pose back, report XY error, stop

Prerequisites (same as the main program):
  - the YOLO detection node must be running (publishing center_data_coords
    and center_data_labels).
  - build / source install so `import hiwin_control.arm_controler_how` picks up
    the latest pixel_mm_convert.
  - run from the workspace root:
      python3 src/hiwin_control/hiwin_control/move_above_ball.py

SAFETY: start from a safe pose, keep the area clear. A slow velocity is used.
"""
import time
from threading import Thread

import rclpy
from rclpy.node import Node
from rclpy.task import Future
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from hiwin_interfaces.srv import RobotCommand

# Reuse the production calibration + math verbatim (single source of truth).
import hiwin_control.arm_controler_how as ctrl

HIT_TOOL = 8        # cue hitting-end TCP: reaches any ball, corners included
PHOTO_TOOL = 0      # armpos was recorded with the flange (tool 0)
ABOVE_Z = -40.0     # base-frame Z to park the hitting end above the ball
VELOCITY = 30       # slow for safety
ACCELERATION = 30

DETECT_TIMEOUT = 10.0   # s to wait for a YOLO detection before aborting


class MoveAboveBall(Node):
    def __init__(self):
        super().__init__('move_above_ball')
        self.client = self.create_client(RobotCommand, 'hiwinmodbus_service')
        self.create_subscription(Float64MultiArray, 'center_data_coords',
                                 self.yolo_callback, 10)
        self.create_subscription(String, 'center_data_labels',
                                 self.label_callback, 10)
        self.all_ball_pose = []
        self.all_label = []
        self.data_flag = 0

    # --- YOLO callbacks (same contract as arm_controler_how) ---
    def yolo_callback(self, msg):
        if not msg.data:
            self.data_flag = 0
        else:
            self.all_ball_pose = msg.data
            self.data_flag = 1

    def label_callback(self, msg):
        self.all_label = eval(msg.data)

    # --- service plumbing (same pattern as move_to_armpos) ---
    def _wait(self, future: Future, timeout=-1):
        t0 = time.time()
        while not future.done():
            time.sleep(0.01)
            if timeout > 0 and time.time() - t0 > timeout:
                self.get_logger().error('service call timeout')
                return False
        return True

    def call(self, req):
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('waiting for hiwinmodbus_service...')
        future = self.client.call_async(req)
        return future.result() if self._wait(future) else None

    def request(self, cmd_mode=RobotCommand.Request.PTP, tool=0, pose=None):
        req = RobotCommand.Request()
        req.cmd_mode = cmd_mode
        req.cmd_type = RobotCommand.Request.POSE_CMD
        req.velocity = VELOCITY
        req.acceleration = ACCELERATION
        req.tool = tool
        req.base = 0
        req.holding = True
        req.pose = pose if pose is not None else Twist()
        return req

    def check_pose(self, tool):
        res = self.call(self.request(cmd_mode=RobotCommand.Request.CHECK_POSE,
                                     tool=tool))
        return list(res.current_position)

    def ptp(self, xyz, rpy, tool):
        pose = Twist()
        [pose.linear.x, pose.linear.y, pose.linear.z] = xyz
        [pose.angular.x, pose.angular.y, pose.angular.z] = rpy
        return self.call(self.request(cmd_mode=RobotCommand.Request.PTP,
                                      pose=pose, tool=tool))

    # --- the sequence ---
    def run_sequence(self):
        # 1) photo pose (flange at armpos)
        self.get_logger().info(f'Moving to photo pose {ctrl.FIX_ABS_CAM} ...')
        self.ptp(ctrl.FIX_ABS_CAM[0:3], ctrl.FIX_ABS_CAM[3:6], PHOTO_TOOL)

        # 2) wait for YOLO, pick the ball nearest the image center
        time.sleep(1.0)  # let detection settle
        t0 = time.time()
        while self.data_flag == 0 or not self.all_ball_pose:
            if time.time() - t0 > DETECT_TIMEOUT:
                self.get_logger().error('No YOLO detection; aborting (no move).')
                return
            time.sleep(0.05)
        balls = list(self.all_ball_pose)
        mid_x, mid_y = ctrl.check_mid_pose(balls)
        self.get_logger().info(
            f'Centre-most ball pixel: ({mid_x:.1f}, {mid_y:.1f})')

        # 3) pixel -> mm -> base-frame XY (one-shot, first photo)
        ball_mm = ctrl.pixel_mm_convert(ctrl.CAM_TO_TABLE, [mid_x, mid_y])
        ball_base = ctrl.convert_arm_pose(ball_mm, ctrl.FIX_ABS_CAM)
        x, y = ball_base[0], ball_base[1]
        self.get_logger().info(f'Ball base-frame XY: ({x:.2f}, {y:.2f})')

        # 4) move the hitting end (tool 8) directly above, keep orientation
        cur = self.check_pose(HIT_TOOL)
        self.get_logger().info(
            f'Moving tool {HIT_TOOL} above ball -> [{x:.2f}, {y:.2f}, {ABOVE_Z}]')
        self.ptp([x, y, ABOVE_Z], cur[3:6], HIT_TOOL)

        # 5) report actual pose + XY error
        actual = self.check_pose(HIT_TOOL)
        self.get_logger().info(f'Done. Tool {HIT_TOOL} actual pose: {actual}')
        self.get_logger().info(
            f'XY error vs target: ({actual[0] - x:.2f}, {actual[1] - y:.2f}) mm')

    def _run(self):
        try:
            self.run_sequence()
        finally:
            self.destroy_node()

    def start_thread(self):
        Thread(target=self._run, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = MoveAboveBall()
    node.start_thread()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
