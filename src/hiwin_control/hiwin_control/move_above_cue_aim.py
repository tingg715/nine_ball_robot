#!/usr/bin/env python3
"""
One-shot helper: park the cue tool (tool 8) directly ABOVE the WHITE (cue) ball,
using the *shot orientation* computed by the strategy -- but WITHOUT pulling back
the ghost-ball distance. It does NOT descend and does NOT hit.

Idea (why this exists):
  The real program (arm_controler_how.py) aims the cue by:
    - taking the shot vector (vx, vy) from table2.main()
    - Rz = yaw_angle(vx, vy) - 90                (HITPOINT_ANGLE)
    - moving to the *pulled-back* hitpoint (cue center - 1.5r)   (HITPOINT_TOP)
  Here we keep the SAME orientation (Rz from the shot vector) but IGNORE the
  pulled-back hitpoint and park straight over the cue-ball center instead.
  So we never touch the strategy's global ball radius `r` -- we just don't use
  the hitpoint position it returns.

  Use this to eye-check XY + aim angle: is the cue, posed as if to shoot, sitting
  right over the white ball, pointing along the shot line?

It reuses the exact production math from arm_controler_how.py (pixel_mm_convert,
convert_arm_pose, yaw_angle, and every calibration constant) plus nine_ball_strat
for the shot direction -- no duplicated / drifting calibration.

Flow:
  1. flange -> FIX_ABS_CAM (armpos) photo pose                     [tool 0]
  2. wait for YOLO, convert every ball pixel -> base-frame XY
  3. find the white ball (cue) from the labels; the rest are object balls
  4. table2.main(object balls, cue) -> shot vector (vx, vy)
  5. Rz = yaw_angle(vx, vy) - 90   (the shot orientation)
  6. tool 8 -> [cuex, cuey, ABOVE_Z] with rpy [ABOVE_RX, ABOVE_RY, Rz]
  7. read the pose back, report XY error, stop

Prerequisites (same as the main program):
  - the YOLO detection node must be running (center_data_coords + labels).
  - the white ball AND at least one object ball must be in view (the strategy
    needs an object ball to compute a shot direction).
  - build / source install so `import hiwin_control.arm_controler_how` picks up
    the latest calibration.
  - run from the workspace root:
      python3 src/hiwin_control/hiwin_control/move_above_cue_aim.py

SAFETY: start from a safe pose, keep the area clear. Slow velocity is used.
The cue tip parks right over the ball (no ghost-ball pullback), so keep ABOVE_Z
high enough that the tip does not collide with the ball.
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
import hiwin_control.nine_ball_strat as table2

HIT_TOOL = 8        # cue tool TCP to park above the ball
PHOTO_TOOL = 0      # armpos was recorded with the flange (tool 0)

# --- pose to park above the cue ball -----------------------------------------
ABOVE_Z = -65.0     # base-frame Z above the ball. Raise if the tip collides.
ABOVE_RX = -180.0   # tool flipped down (keep)
ABOVE_RY = 0.0      # pitch. Start at 0; if unreachable / tip collides, try ~20
                    # (main program uses 20 for obstacle pitch). Range ~ +-30.
# Rz is computed from the shot vector at runtime (yaw - 90).

X_COMP = 10.0       # mm added to the commanded target X (systematic-error fix)
Y_COMP = 0.0        # mm added to the commanded target Y

VELOCITY = 30       # slow for safety
ACCELERATION = 30

DETECT_TIMEOUT = 10.0   # s to wait for a YOLO detection before aborting
SETTLE_TIME = 1.5   # s to let YOLO settle after the arm parks at the photo pose
AVG_FRAMES = 5      # fresh, label-consistent frames to average for stable XY
REACH_TOL = 15.0    # mm: if actual XY is farther than this from the commanded
                    # target, treat the move as failed (out of reach / IK fail)


class MoveAboveCueAim(Node):
    def __init__(self):
        super().__init__('move_above_cue_aim')
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

    # --- service plumbing (same pattern as move_above_ball) ---
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

    def capture_detection(self):
        """Grab a stable detection while parked at the photo pose.

        1) sleep SETTLE_TIME so YOLO settles and any in-flight (mid-motion) frame
           is flushed;
        2) discard the current frame (force data_flag=0) so we only accept frames
           that arrive strictly AFTER settling -- never a stale one;
        3) collect up to AVG_FRAMES fresh frames that all report the SAME labels
           (flickering frames with a different ball set are skipped) and average
           their pixel coords, so one bad frame can't skew XY.

        Returns (averaged_coords, labels) or (None, None) on timeout.
        """
        time.sleep(SETTLE_TIME)
        self.data_flag = 0
        frames = []
        ref_labels = None
        t0 = time.time()
        while len(frames) < AVG_FRAMES:
            if time.time() - t0 > DETECT_TIMEOUT:
                break
            if self.data_flag == 0 or not self.all_ball_pose:
                time.sleep(0.02)
                continue
            coords = list(self.all_ball_pose)
            labels = list(self.all_label)
            self.data_flag = 0          # consume; wait for the next fresh frame
            if len(labels) * 2 != len(coords):
                continue                # labels/coords out of sync -> skip
            if ref_labels is None:
                ref_labels = labels
            if labels != ref_labels:
                continue                # ball set changed (flicker) -> skip
            frames.append(coords)
        if not frames:
            return None, None
        avg = [sum(f[i] for f in frames) / len(frames)
               for i in range(len(frames[0]))]
        self.get_logger().info(
            f'Averaged {len(frames)} frame(s); labels {ref_labels}')
        return avg, ref_labels

    # --- the sequence ---
    def run_sequence(self):
        # 1) photo pose (flange at armpos)
        self.get_logger().info(f'Moving to photo pose {ctrl.FIX_ABS_CAM} ...')
        self.ptp(ctrl.FIX_ABS_CAM[0:3], ctrl.FIX_ABS_CAM[3:6], PHOTO_TOOL)

        # 2) capture a stable detection (settle + discard stale + average frames)
        balls, labels = self.capture_detection()
        if balls is None:
            self.get_logger().error('No stable YOLO detection; aborting (no move).')
            return
        n = len(balls) // 2

        # 2b) pixel -> mm -> base-frame XY for every ball (one-shot, first photo)
        base_x, base_y = [], []
        for i in range(0, 2 * n, 2):
            mm = ctrl.pixel_mm_convert(ctrl.CAM_TO_TABLE, balls[i:i + 2])
            base = ctrl.convert_arm_pose(mm, ctrl.FIX_ABS_CAM)
            base_x.append(base[0])
            base_y.append(base[1])

        # 3) find the white (cue) ball; everything else is an object ball
        if 'white' not in labels:
            self.get_logger().error('No white (cue) ball detected; aborting.')
            return
        cue_idx = labels.index('white')
        cuex, cuey = base_x[cue_idx], base_y[cue_idx]
        objectx = [base_x[i] for i in range(n) if i != cue_idx]
        objecty = [base_y[i] for i in range(n) if i != cue_idx]
        if not objectx:
            self.get_logger().error(
                'Only the cue ball is in view; strategy needs an object ball. '
                'Aborting.')
            return
        self.get_logger().info(f'Cue (white) base-frame XY: ({cuex:.2f}, {cuey:.2f})')

        # 4) strategy -> shot vector (we use ONLY vx, vy; we drop its hitpoint)
        #    table2.main -> [score, vx, vy, obstacle_flag, hitpointx, hitpointy]
        info = table2.main(objectx, objecty, cuex, cuey)
        score, vx, vy = info[0], info[1], info[2]
        self.get_logger().info(f'Shot vector (vx, vy) = ({vx:.2f}, {vy:.2f})')

        # 5) shot orientation: Rz = yaw - 90 (same as HITPOINT_ANGLE)
        yaw, _ = ctrl.yaw_angle(vx, vy)
        rz = yaw - 90.0
        rpy = [ABOVE_RX, ABOVE_RY, rz]
        self.get_logger().info(f'Aim rpy = {rpy}')

        # 6) park tool 8 straight above the cue ball (NO ghost-ball pullback)
        #    apply the systematic XY compensation to the commanded target
        target_x = cuex + X_COMP
        target_y = cuey + Y_COMP
        self.get_logger().info(
            f'Moving tool {HIT_TOOL} above cue ball -> '
            f'[{target_x:.2f}, {target_y:.2f}, {ABOVE_Z}]  rpy {rpy}  '
            f'(comp {X_COMP:+.1f}, {Y_COMP:+.1f})')
        res = self.ptp([target_x, target_y, ABOVE_Z], rpy, HIT_TOOL)
        if res is not None and res.arm_state != RobotCommand.Response.IDLE:
            self.get_logger().warn(f'Move returned arm_state={res.arm_state} (not IDLE).')

        # 7) verify: compare actual pose to the commanded target
        actual = self.check_pose(HIT_TOOL)
        ex, ey = actual[0] - target_x, actual[1] - target_y
        self.get_logger().info(f'Done. Tool {HIT_TOOL} actual pose: {actual}')
        if abs(ex) > REACH_TOL or abs(ey) > REACH_TOL:
            self.get_logger().error(
                f'Arm did NOT reach the target (off by {ex:.1f}, {ey:.1f} mm). '
                f'The pose [{target_x:.1f}, {target_y:.1f}, {ABOVE_Z}] rpy {rpy} is most '
                f'likely OUT OF REACH / IK-infeasible. This is NOT a calibration '
                f'error -- try a ball closer to the arm, raise ABOVE_Z, or give '
                f'ABOVE_RY some pitch.')
        else:
            self.get_logger().info(
                f'XY error vs cue center: ({ex:.2f}, {ey:.2f}) mm')

    def _run(self):
        try:
            self.run_sequence()
        finally:
            self.destroy_node()

    def start_thread(self):
        Thread(target=self._run, daemon=True).start()


def main(args=None):
    rclpy.init(args=args)
    node = MoveAboveCueAim()
    node.start_thread()
    rclpy.spin(node)
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
