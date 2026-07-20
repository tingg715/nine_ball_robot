#!/usr/bin/env python3
"""
Hiwin Ball Position Accuracy Test UI

KEY FIXES vs previous version:
  1. Threading deadlock removed — arm state machine uses an Event instead of
     calling rclpy.spin_once() (which conflicts with the spin thread).
  2. State flow corrected — MOVE_TO_PHOTO_POSE → LOCK_INFO → DYNAMIC_CALI
     still runs so updated_target_cue is always populated before STRATEGY.
     UI waits at WAIT_FOR_UI *after* calibration, not before it.
  3. Coordinate fix — UI canvas coords are converted to arm-frame mm before
     being stored in strategy_info[4]/[5].
  4. Auto / Manual toggle button added to UI.
  5. eval() replaced with json.loads() for safety.
  6. Null-check added on every call_hiwin() result.
  7. Busy-wait loops replaced with threading.Event.

TEST BEHAVIOR:
  Select a target ball in the UI, then move the cue tool to a safe Z above it.
  No second calibration, no descent to hit height, and no digital-output hit.

Run:
  ros2 run hiwin_control arm_controller
"""

import time, yaml, configparser, os, math, threading, json, signal
import tkinter as tk
from tkinter import ttk, messagebox

import rclpy
from enum import Enum
from threading import Thread, Event
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray, String
from hiwin_interfaces.srv import RobotCommand

import numpy as np
import quaternion as qtn
import hiwin_control.transformations as transformations
import hiwin_control.nine_ball_strat as table2

# ═══════════════════════════════════════════════════
#  Hardware constants
# ═══════════════════════════════════════════════════
CUE_TOOL         = 9
DEFAULT_VELOCITY = 70
DEFAULT_ACCEL    = 70
LIGHT_PIN        = 6
HITSOFT_PIN      = 4
HITMID_PIN       = 5
HITHEAVY_PIN     = 1
HEAVY_PIN        = 2
END_TURN_RIGHT   = [90.00, 0.00, 0.00, 0.00, -90.00, 0.00]
TOOL_TO_CAM      = [-35.5467, 77.6499, -68.2193]

# Hover height for calibration
CALI_Z           = 85.0
# Z heights for hit sequence (mm in arm frame)
HIT_TOP_Z        = -70.0
HIT_FLAT_Z       = -135.0
HIT_OBSTACLE_Z   = -122.445
PITCH_Z          = 132.0
PITCH_Y          = 408.0
PITCH_X          = 0.0
PITCH_ANGLE_Y    = 20.0

# Safe Z height used by the position-only test mode.
# Start high and lower gradually only after confirming X/Y alignment.
TEST_ABOVE_Z     = 0.0

# ── Load calibration files ──────────────────────────
_cwd = os.getcwd()
_cfg = configparser.ConfigParser()
_cfg.read(_cwd + '/src/hiwin_control/hiwin_control/eye_in_hand_calibration.ini')
_tm = _cfg['hand_eye_calibration']
TOOL_TO_CAM[0] = float(_tm['y']) * 1000
TOOL_TO_CAM[1] = float(_tm['x']) * 1000
TOOL_TO_CAM[2] = -float(_tm['z']) * 1000

_camera_cfg_path = (
    '/home/ting/work/src/hiwin_control/hiwin_control/camera_calibration.ini')
if not os.path.isfile(_camera_cfg_path):
    raise RuntimeError(
        f'Camera calibration file does not exist: {_camera_cfg_path}')

_camera_cfg = configparser.ConfigParser()
try:
    with open(_camera_cfg_path, encoding='utf-8') as _camera_cfg_file:
        _camera_cfg.read_file(_camera_cfg_file)
except (OSError, configparser.Error) as exc:
    raise RuntimeError(
        f'Failed to read camera calibration file {_camera_cfg_path}: {exc}') from exc

_camera_section = 'Intrinsic'
if not _camera_cfg.has_section(_camera_section):
    raise RuntimeError(
        f'Camera calibration file {_camera_cfg_path} is missing '
        f'[{_camera_section}] section')

_camera_fields = {'fx': '0_0', 'fy': '1_1', 'cx': '0_2', 'cy': '1_2'}
_camera_values = {}
for _name, _field in _camera_fields.items():
    if not _camera_cfg.has_option(_camera_section, _field):
        raise RuntimeError(
            f'Camera calibration file {_camera_cfg_path} is missing '
            f'[{_camera_section}] field {_field} ({_name})')
    try:
        _value = float(_camera_cfg.get(_camera_section, _field))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f'Invalid numeric value for [{_camera_section}] {_field} '
            f'({_name}) in {_camera_cfg_path}') from exc
    if not math.isfinite(_value):
        raise RuntimeError(
            f'Non-finite value for [{_camera_section}] {_field} '
            f'({_name}) in {_camera_cfg_path}: {_value}')
    _camera_values[_name] = _value

CAMERA_FX = _camera_values['fx']
CAMERA_FY = _camera_values['fy']
CAMERA_CX = _camera_values['cx']
CAMERA_CY = _camera_values['cy']
if CAMERA_FX <= 0 or CAMERA_FY <= 0:
    raise RuntimeError(
        f'Camera focal lengths must be positive in {_camera_cfg_path}: '
        f'fx={CAMERA_FX}, fy={CAMERA_FY}')

with open(_cwd + '/src/hiwin_control/hiwin_control/arm.yaml') as _f:
    _arm = yaml.safe_load(_f)
FIX_ABS_CAM       = _arm['armpos']
FIX_ABS_RIGHT_CAM = [FIX_ABS_CAM[0]+150, 376.998, 411.897, 180.0, 0.0, 90.0]
FIX_ABS_LEFT_CAM  = [FIX_ABS_CAM[0]-150, 376.998, 411.897, 180.0, 0.0, 90.0]
CAM_TO_TABLE      = _arm['zoff']

# ── Camera ↔ table homography ───────────────────────
_img_pots = [_arm['img_pot0'], _arm['img_pot1'],
             _arm['img_pot2'], _arm['img_pot3']]
_arm_pots = [_arm['pot0'][:2], _arm['pot1'][:2],
             _arm['pot2'][:2], _arm['pot3'][:2]]

def _compute_homography(src, dst):
    """4-point DLT homography (numpy only, no OpenCV needed)."""
    A = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        A.extend([[-sx,-sy,-1,  0,  0,  0, dx*sx, dx*sy, dx],
                  [  0,  0,  0,-sx,-sy,-1, dy*sx, dy*sy, dy]])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def _apply_h(H, x, y):
    p = H @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])

_H_IMG_TO_ARM = _compute_homography(_img_pots, _arm_pots)
_TBL_X0 = min(p[0] for p in _arm_pots)
_TBL_X1 = max(p[0] for p in _arm_pots)
_TBL_Y0 = min(p[1] for p in _arm_pots)
_TBL_Y1 = max(p[1] for p in _arm_pots)

def yolo_to_canvas(px, py):
    """YOLO image pixel → canvas pixel via homography + table mapping."""
    ax, ay = _apply_h(_H_IMG_TO_ARM, px, py)
    return arm_mm_to_canvas(ax, ay)

def arm_mm_to_canvas(ax, ay):
    """Arm-frame mm → canvas pixel."""
    cx = UPAD + (ax - _TBL_X0) / (_TBL_X1 - _TBL_X0) * UTW
    cy = UPAD + (_TBL_Y1 - ay) / (_TBL_Y1 - _TBL_Y0) * UTH
    return cx, cy

def canvas_to_arm_mm(cx, cy, arm_ref_pose=None):
    """Canvas pixel → arm-frame mm (direct inverse, no camera chain)."""
    ax = _TBL_X0 + (cx - UPAD) / UTW * (_TBL_X1 - _TBL_X0)
    ay = _TBL_Y1 - (cy - UPAD) / UTH * (_TBL_Y1 - _TBL_Y0)
    return ax, ay

# ═══════════════════════════════════════════════════
#  UI layout constants
# ═══════════════════════════════════════════════════
# Table real dimensions (mm) — 1 px ≈ 1 mm
TABLE_WIDTH_MM  = 627
TABLE_HEIGHT_MM = 304
UPAD    = 36
UTW     = TABLE_WIDTH_MM    # 627 px
UTH     = TABLE_HEIGHT_MM   # 304 px
UW      = UTW + UPAD * 2    # 699
UH      = UTH + UPAD * 2    # 376
BR      = 16    # ball radius mm  (30.4cm:1.6cm = 932px:97.5px)
PR      = 20    # pocket radius mm (62.6cm:2cm = 1920px:60px)
YOLO_W, YOLO_H = 1920, 1080

# Pocket canvas positions derived from real arm-frame coordinates
_pc   = [arm_mm_to_canvas(x, y) for x, y in _arm_pots]
_mx   = (_arm_pots[0][0] + _arm_pots[1][0]) / 2   # mid-pocket X ≈ 0
_mty  = (_arm_pots[0][1] + _arm_pots[1][1]) / 2   # top-mid Y
_mby  = (_arm_pots[2][1] + _arm_pots[3][1]) / 2   # bot-mid Y
POCKET_POSITIONS = [
    (*_pc[0], 'TL', 'Top-Left'),
    (*arm_mm_to_canvas(_mx, _mty), 'TM', 'Top-Mid'),
    (*_pc[1], 'TR', 'Top-Right'),
    (*_pc[3], 'BL', 'Bot-Left'),
    (*arm_mm_to_canvas(_mx, _mby), 'BM', 'Bot-Mid'),
    (*_pc[2], 'BR', 'Bot-Right'),
]

BALL_COLORS = {
    'white':'#ffffff','cue':'#ffffff','w':'#ffffff',
    '1':'#f5e642','2':'#1a3ab5','3':'#e74c3c',
    '4':'#6e2da8','5':'#e67e22','6':'#1a7a3a',
    '7':'#8b1a1a','8':'#222222','9':'#f5e642',
}

UI_WATCH   = 'WATCHING'
UI_SELECT  = 'SELECT_TARGET'
UI_PREVIEW = 'PREVIEW'
UI_HITTING = 'HITTING'

# ═══════════════════════════════════════════════════
#  Arm states
# ═══════════════════════════════════════════════════
class States(Enum):
    INIT            = 0
    FINISH          = 1
    MOVE_TO_PHOTO   = 2
    DYNAMIC_CALI    = 3
    STRATEGY        = 4
    HITPOINT_TOP    = 5
    HITPOINT_ANGLE  = 6
    HITBALL_POSE    = 7
    HITBALL         = 8
    AF_HIT_TOP      = 9
    CHECK_POSE      = 10
    OPEN_SEC_IO     = 11
    LOCK_INFO       = 12
    FIX_RIGHT       = 13
    FIX_LEFT        = 14
    LOCK_CUE        = 15
    HITPOINT_PITCH  = 16
    WAIT_FOR_UI     = 17
    TEST_MOVE_ABOVE = 18


# ═══════════════════════════════════════════════════
#  Geometry helpers
# ═══════════════════════════════════════════════════
def check_mid_pose(all_ball_pose):
    errors = [math.hypot(all_ball_pose[i]-CAMERA_CX,
                         all_ball_pose[i+1]-CAMERA_CY)
              for i in range(0, len(all_ball_pose), 2)]
    idx = errors.index(min(errors))
    return all_ball_pose[2*idx], all_ball_pose[2*idx+1]

def mid_point_error(mid_ball):
    return math.hypot(mid_ball[0]-CAMERA_CX, mid_ball[1]-CAMERA_CY)

def pixel_mm_convert(cam_h, pixels):
    if pixels is None:
        raise ValueError('pixels must not be None')
    try:
        if len(pixels) < 2:
            raise ValueError('pixels must contain at least two values: [u, v]')
    except TypeError as exc:
        raise TypeError('pixels must be a sequence containing [u, v]') from exc

    try:
        u = float(pixels[0])
        v = float(pixels[1])
        z_mm = float(cam_h)
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError('u, v, and cam_h must be convertible to float') from exc

    if not all(math.isfinite(value) for value in (u, v, z_mm)):
        raise ValueError('u, v, and cam_h must be finite numbers')
    if z_mm <= 0:
        raise ValueError(f'cam_h must be greater than 0, got {z_mm}')
    if CAMERA_FX <= 0 or CAMERA_FY <= 0:
        raise RuntimeError(
            f'Camera focal lengths must be positive, got '
            f'fx={CAMERA_FX}, fy={CAMERA_FY}')

    x_mm = (u - CAMERA_CX) * z_mm / CAMERA_FX
    y_mm = (v - CAMERA_CY) * z_mm / CAMERA_FY
    return [x_mm, y_mm, 1.0]

def yaw_angle(vx, vy):
    vlen = math.hypot(vx, vy)
    if vlen == 0:
        return 0.0, 0.0
    rad = math.acos((-vy) / vlen)
    return (rad*180/math.pi, rad) if vx >= 0 else (-rad*180/math.pi, -rad)

def convert_arm_pose(ball_pose, arm_pose):
    base2tool = np.array(arm_pose, dtype=float)
    base2tool[:3] /= 1000
    tool2cam_q = [0.007269923959234842, -0.0032559856852129544,
                  0.7024893071973962,    0.7116497172318463]
    tool2cam_t = [0.07750464277165502, -0.03671563427202898, 0.06849008934693676]
    tool2cam_rot = qtn.as_rotation_matrix(
        np.quaternion(tool2cam_q[3], *tool2cam_q[:3]))
    tool2cam_mat = np.append(tool2cam_rot, np.array([tool2cam_t]).T, axis=1)
    tool2cam_mat = np.vstack([tool2cam_mat, [0,0,0,1]])

    quat = transformations.quaternion_from_euler(
        base2tool[3]*math.pi/180, base2tool[4]*math.pi/180,
        base2tool[5]*math.pi/180, axes='sxyz')
    b2t_rot = qtn.as_rotation_matrix(np.quaternion(quat[3], *quat[:3]))
    b2t_mat = np.append(b2t_rot, np.array([base2tool[:3]]).T, axis=1)
    b2t_mat = np.vstack([b2t_mat, [0,0,0,1]])

    c2b_t = np.array([[ball_pose[0]/1000, ball_pose[1]/1000, ball_pose[2]]])
    c2b_mat = np.append(np.identity(3), c2b_t.T, axis=1)
    c2b_mat = np.vstack([c2b_mat, [0,0,0,1]])

    base2ball = b2t_mat @ tool2cam_mat @ c2b_mat
    ax, ay, az = transformations.euler_from_matrix(base2ball)
    t = transformations.translation_from_matrix(base2ball) * 1000
    return [t[0], t[1], t[2], ax*180/math.pi, ay*180/math.pi, az*180/math.pi]


# ═══════════════════════════════════════════════════
#  UI physics helpers
# ═══════════════════════════════════════════════════
def ui_simulate_path(sx, sy, vx, vy, steps, bounce=True):
    path = [(sx, sy)]
    x, y = sx, sy
    for _ in range(steps):
        x += vx; y += vy
        if bounce:
            if x <= UPAD+BR:    vx =  abs(vx); x = UPAD+BR
            if x >= UW-UPAD-BR: vx = -abs(vx); x = UW-UPAD-BR
            if y <= UPAD+BR:    vy =  abs(vy); y = UPAD+BR
            if y >= UH-UPAD-BR: vy = -abs(vy); y = UH-UPAD-BR
        else:
            if not (UPAD <= x <= UW-UPAD and UPAD <= y <= UH-UPAD):
                break
        path.append((x, y))
    return path

def ui_collision(wvx, wvy, hx, hy, tx, ty):
    dx, dy = tx-hx, ty-hy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return (wvx, wvy), (0.0, 0.0)
    nx, ny = dx/dist, dy/dist
    dot = wvx*nx + wvy*ny
    tvx, tvy = dot*nx, dot*ny
    return (tvx, tvy), (wvx-tvx, wvy-tvy)

def ui_calc_hit_angle(wx, wy, tx, ty, px, py):
    t2p = math.atan2(py-ty, px-tx)
    gx = tx - math.cos(t2p)*BR*2
    gy = ty - math.sin(t2p)*BR*2
    angle = math.degrees(math.atan2(gy-wy, gx-wx)) % 360
    return angle, gx, gy



# ═══════════════════════════════════════════════════
#  ROS2 Node
# ═══════════════════════════════════════════════════
class Hiwin_Controller(Node):

    def __init__(self):
        super().__init__('hiwin_ball_position_test')

        self.hiwin_client = self.create_client(RobotCommand, 'hiwinmodbus_service')
        self.yolo_sub  = self.create_subscription(
            Float64MultiArray, 'center_data_coords', self._yolo_cb, 10)
        self.label_sub = self.create_subscription(
            String, 'center_data_labels', self._label_cb, 10)

        # ── Shared state (protected by _lock) ──
        self._lock            = threading.Lock()
        self.all_ball_pose    = []
        self.all_label        = []
        self.data_flag        = 0
        self._yolo_active     = True    # test UI always accepts YOLO detections

        # arm-loop local
        self.ball_pose_buffer   = []
        self.label_buffer       = []
        self.ball_pose          = []
        self.target_cue         = []
        self.updated_target_cue = []
        self.strategy_info      = []
        self.index              = 0
        self.fix_z              = CALI_Z
        self.table_z            = FIX_ABS_CAM[2] + TOOL_TO_CAM[2] - CAM_TO_TABLE
        self.fix_check_point    = []
        self.current_pose       = [0]*6
        self.current_tool_pose  = [0]*6
        self.hitpointx          = 0.0
        self.hitpointy          = 0.0
        self.score              = 0
        self.obstacle           = 0

        # UI shot handoff — arm blocks on this event
        self._shot_event   = Event()
        self._reset_event  = Event()
        # UI is created only after the arm reaches arm.yaml -> armpos.
        self.photo_pose_ready = Event()
        self.startup_failed   = Event()
        self.ui_shot_info  = None   # [arm_x_mm, arm_y_mm, angle_deg, force_1_10]
        self.ui_white_yolo = None   # [px, py] white ball YOLO pixel from UI freeze
        self.ui_target_yolo= None   # [px, py] target ball YOLO pixel from UI freeze

        # Mode: True = wait for UI; False = auto table2.main()
        self.use_ui_mode  = True
        self._yolo_msg_count = 0
        self._printed_ball_coordinates = False

        self.get_logger().info('Hiwin ball-position test ready')
        self.get_logger().info('Subscribing: /center_data_coords and /center_data_labels')

    # ── ROS callbacks ─────────────────────────────
    def _yolo_cb(self, msg):
        with self._lock:
            # Position-test mode always accepts YOLO data so the UI can display
            # balls even while the arm is moving or waiting for its service.
            if msg.data:
                self.all_ball_pose = list(msg.data)
                self._yolo_msg_count += 1
                if self._yolo_msg_count == 1 or self._yolo_msg_count % 100 == 0:
                    self.get_logger().info(
                        f'YOLO coords received: {len(self.all_ball_pose)//2} balls')
                self.target_cue = [list(self.all_ball_pose[:2]),
                                   list(self.all_ball_pose[-2:])]
                self.data_flag = 1
            else:
                self.data_flag = 0

    def _label_cb(self, msg):
        try:
            labels = json.loads(msg.data)
        except Exception:
            try:
                labels = list(eval(msg.data))   # fallback if publisher sends repr
            except Exception:
                labels = []
        with self._lock:
            self.all_label = labels

    # ── UI handoff ────────────────────────────────
    def request_photo_pose(self):
        """Called from UI Reset — unblocks WAIT_FOR_UI and sends arm back to photo pose."""
        self._reset_event.set()
        self.get_logger().info('Reset requested: returning to photo pose')

    def set_shot_from_ui(self, arm_x_mm, arm_y_mm, angle_deg, force,
                         white_yolo=None, target_yolo=None):
        """Called from UI thread. Unblocks the arm state machine."""
        with self._lock:
            self.ui_shot_info   = [arm_x_mm, arm_y_mm, angle_deg, force]
            self.ui_white_yolo  = white_yolo   # [px, py] for LOCK_INFO re-detect
            self.ui_target_yolo = target_yolo
        self._shot_event.set()
        self.get_logger().info(
            f'Shot from UI: x={arm_x_mm:.1f} y={arm_y_mm:.1f} '
            f'angle={angle_deg:.1f}° force={force}')

    def get_current_pose(self):
        """Non-blocking: returns the last known pose, or queries if arm is idle."""
        with self._lock:
            return list(self.current_pose)

    def get_balls_for_ui(self):
        with self._lock:
            coords = list(self.all_ball_pose)
            labels = list(self.all_label)
        balls = []
        for i in range(0, len(coords)-1, 2):
            idx   = i // 2
            label = labels[idx] if idx < len(labels) else str(idx)
            pixel_u = coords[i]
            pixel_v = coords[i+1]
            arm_x, arm_y = _apply_h(_H_IMG_TO_ARM, pixel_u, pixel_v)
            canvas_x, canvas_y = arm_mm_to_canvas(arm_x, arm_y)
            balls.append({
                'label': label,
                'x': canvas_x,
                'y': canvas_y,
                'yolo_x': pixel_u,
                'yolo_y': pixel_v,
                'arm_x': arm_x,
                'arm_y': arm_y,
            })

        if balls and not self._printed_ball_coordinates:
            for ball in balls:
                self.get_logger().info(
                    f'Ball {ball["label"]}: '
                    f'YOLO=({ball["yolo_x"]:.2f}, {ball["yolo_y"]:.2f}), '
                    f'Homography Base=({ball["arm_x"]:.2f}, '
                    f'{ball["arm_y"]:.2f}), '
                    f'Canvas=({ball["x"]:.2f}, {ball["y"]:.2f})')
            self._printed_ball_coordinates = True
        return balls

    # ── Arm state machine ─────────────────────────
    def _sm(self, state):
        if state == States.INIT:
            # Position-test mode does not move to END_TURN_RIGHT.
            # Go directly to the photo pose loaded from arm.yaml.
            self.get_logger().info('Starting directly with arm.yaml photo pose')
            return States.MOVE_TO_PHOTO

        elif state == States.MOVE_TO_PHOTO:
            with self._lock:
                self._yolo_active = True
                self._printed_ball_coordinates = False
            self.updated_target_cue = []
            self.index = 0
            # Position-test startup: move first; do not send lighting I/O here.
            pose = self._twist(*FIX_ABS_CAM)
            self.get_logger().info(f'Moving directly to photo pose: {FIX_ABS_CAM}')
            res = self._call(self._req(cmd_mode=RobotCommand.Request.PTP, pose=pose))
            if res is None or res.arm_state != RobotCommand.Response.IDLE:
                self.get_logger().error('Failed to reach photo pose; UI will not open')
                self.startup_failed.set()
                return None

            # Notify main() that it is now safe to create the UI.
            self.photo_pose_ready.set()
            self.get_logger().info('Photo pose reached; opening UI')
            return States.WAIT_FOR_UI

        elif state == States.LOCK_INFO:
            with self._lock:
                self._yolo_active = True
            time.sleep(1.0)
            self.ball_pose = []
            with self._lock:
                self.ball_pose_buffer = list(self.all_ball_pose)
                self.label_buffer     = list(self.all_label)
                white_yolo  = self.ui_white_yolo
                target_yolo = self.ui_target_yolo

            # Find white ball: prefer the UI-selected pixel, fall back to label
            white_px = None
            if white_yolo:
                # match closest YOLO detection to UI-frozen white ball pixel
                best, best_d = None, float('inf')
                for i in range(0, len(self.ball_pose_buffer)-1, 2):
                    d = math.hypot(self.ball_pose_buffer[i]   - white_yolo[0],
                                   self.ball_pose_buffer[i+1] - white_yolo[1])
                    if d < best_d:
                        best_d = d; best = i
                if best is not None:
                    white_px = self.ball_pose_buffer[best:best+2]
            if white_px is None and 'white' in self.label_buffer:
                idx = self.label_buffer.index('white')
                white_px = self.ball_pose_buffer[idx*2:idx*2+2]

            # Find target ball: match UI-selected pixel
            target_px = None
            if target_yolo:
                best, best_d = None, float('inf')
                for i in range(0, len(self.ball_pose_buffer)-1, 2):
                    d = math.hypot(self.ball_pose_buffer[i]   - target_yolo[0],
                                   self.ball_pose_buffer[i+1] - target_yolo[1])
                    if d < best_d:
                        best_d = d; best = i
                if best is not None:
                    target_px = self.ball_pose_buffer[best:best+2]

            # Fall back to first/last if detection failed
            if white_px  is None: white_px  = self.ball_pose_buffer[-2:]
            if target_px is None: target_px = self.ball_pose_buffer[:2]

            self.target_cue = [target_px, white_px]
            for tgt in self.target_cue:
                mm   = pixel_mm_convert(CAM_TO_TABLE, tgt)
                pose = convert_arm_pose(mm, FIX_ABS_CAM)
                self.ball_pose.append(pose[:2])

            if white_px:
                return States.DYNAMIC_CALI
            return States.FIX_RIGHT

        elif state == States.FIX_RIGHT:
            self.fix_check_point = list(FIX_ABS_RIGHT_CAM)
            self._light(True)
            pose = self._twist(*self.fix_check_point)
            res = self._call(self._req(cmd_mode=RobotCommand.Request.PTP, pose=pose))
            if res is None or res.arm_state != RobotCommand.Response.IDLE:
                return None
            return States.LOCK_CUE

        elif state == States.FIX_LEFT:
            self.fix_check_point = list(FIX_ABS_LEFT_CAM)
            self._light(True)
            pose = self._twist(*self.fix_check_point)
            res = self._call(self._req(cmd_mode=RobotCommand.Request.PTP, pose=pose))
            if res is None or res.arm_state != RobotCommand.Response.IDLE:
                return None
            return States.LOCK_CUE

        elif state == States.LOCK_CUE:
            time.sleep(0.3)
            with self._lock:
                self.ball_pose_buffer = list(self.all_ball_pose)
                self.label_buffer     = list(self.all_label)
            if 'white' in self.label_buffer:
                mm     = pixel_mm_convert(CAM_TO_TABLE, self.ball_pose_buffer[-2:])
                actual = convert_arm_pose(mm, self.fix_check_point)
                self.ball_pose[1] = actual[:2]
                self.fix_check_point = list(FIX_ABS_CAM)
                return States.DYNAMIC_CALI
            return States.FIX_LEFT

        elif state == States.DYNAMIC_CALI:
            Kp = 0.25
            self._light(False)
            if self.index >= len(self.target_cue):
                return States.OPEN_SEC_IO

            pose = self._twist(
                self.ball_pose[self.index][0] - TOOL_TO_CAM[0],
                self.ball_pose[self.index][1] - TOOL_TO_CAM[1],
                self.fix_z,
                *FIX_ABS_CAM[3:])
            self._call(self._req(cmd_mode=RobotCommand.Request.PTP, pose=pose))

            res = self._call(self._req(cmd_mode=RobotCommand.Request.CHECK_POSE))
            if res is None:
                return None
            second_photo = list(res.current_position)

            # P-controller to centre ball in frame
            MAX_ITER = 50
            for _ in range(MAX_ITER):
                with self._lock:
                    bp = list(self.all_ball_pose)
                if not bp:
                    time.sleep(0.05); continue
                mid_x, mid_y = check_mid_pose(bp)
                err = mid_point_error([mid_x, mid_y])
                rel = pixel_mm_convert(
                    self.fix_z - abs(TOOL_TO_CAM[2]) + abs(self.table_z),
                    [mid_x, mid_y])
                pose.linear.x = second_photo[0] + Kp*rel[0]
                pose.linear.y = second_photo[1] - Kp*rel[1]
                self._call(self._req(cmd_mode=RobotCommand.Request.PTP, pose=pose))
                second_photo[0] += Kp*rel[0]
                second_photo[1] -= Kp*rel[1]
                if err <= 3:
                    break
                time.sleep(0.02)

            # record calibrated position
            with self._lock:
                bp = list(self.all_ball_pose)
            ball_rel = pixel_mm_convert(
                self.fix_z - abs(TOOL_TO_CAM[2]) + abs(self.table_z),
                self.target_cue[self.index])
            res = self._call(self._req(cmd_mode=RobotCommand.Request.CHECK_POSE))
            if res is None:
                return None
            upd = res.current_position
            self.updated_target_cue.append(upd[0] + TOOL_TO_CAM[0] + ball_rel[0])
            self.updated_target_cue.append(upd[1] + TOOL_TO_CAM[1] - ball_rel[1])
            self.index += 1

            if self.index < len(self.target_cue):
                return States.DYNAMIC_CALI
            return States.OPEN_SEC_IO

        elif state == States.OPEN_SEC_IO:
            with self._lock:
                self._yolo_active = False
            req = self._req(cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                            digital_output_cmd=RobotCommand.Request.DIGITAL_ON,
                            digital_output_pin=HEAVY_PIN)
            res = self._call(req)
            if res is None or res.arm_state != RobotCommand.Response.IDLE:
                return None
            return States.HITPOINT_PITCH

        elif state == States.WAIT_FOR_UI:
            with self._lock:
                self._yolo_active = True
            self.get_logger().info('Waiting for UI shot selection...')
            self._shot_event.clear()
            self._reset_event.clear()
            # Block until UI sends a shot or requests reset
            while True:
                if self._shot_event.wait(timeout=0.1):
                    break
                if self._reset_event.is_set():
                    self.get_logger().info('Reset: going back to photo pose')
                    return States.MOVE_TO_PHOTO
            with self._lock:
                info = list(self.ui_shot_info)
            arm_x, arm_y, angle_deg, force = info
            vx = math.cos(math.radians(angle_deg))
            vy = math.sin(math.radians(angle_deg))
            self.strategy_info = [force * 500, vx, vy, 0, arm_x, arm_y]
            self.get_logger().info(
                f'Position test target received: x={arm_x:.1f} y={arm_y:.1f}')
            # Position-only test: skip LOCK_INFO and DYNAMIC_CALI.
            return States.TEST_MOVE_ABOVE

        elif state == States.TEST_MOVE_ABOVE:
            # Position-only verification mode:
            # move the cue tool above the UI-selected target and stop.
            self._light(False)

            if len(self.strategy_info) < 6:
                self.get_logger().error('Missing test target coordinates')
                return None

            target_x = float(self.strategy_info[4])
            target_y = float(self.strategy_info[5])

            # Read the current cue-tool orientation and preserve it.
            res = self._call(self._req(
                cmd_mode=RobotCommand.Request.CHECK_POSE,
                tool=CUE_TOOL))
            if res is None:
                self.get_logger().error('Cannot read cue tool pose')
                return None

            current_tool_pose = list(res.current_position)

            pose = self._twist(
                target_x, target_y, TEST_ABOVE_Z,
                current_tool_pose[3],
                current_tool_pose[4],
                current_tool_pose[5])

            self.get_logger().info(
                '[TEST_MOVE_ABOVE command]\n'
                f'Tool={CUE_TOOL} Base=0\n'
                'Target pose:\n'
                f'X={target_x:.2f}\n'
                f'Y={target_y:.2f}\n'
                f'Z={TEST_ABOVE_Z:.2f}\n'
                f'Rx={current_tool_pose[3]:.2f}\n'
                f'Ry={current_tool_pose[4]:.2f}\n'
                f'Rz={current_tool_pose[5]:.2f}\n'
                'Velocity=20\n'
                'Acceleration=20')

            res = self._call(self._req(
                cmd_mode=RobotCommand.Request.PTP,
                tool=CUE_TOOL,
                pose=pose,
                velocity=20,
                acceleration=20))

            if res is None:
                self.get_logger().error('Position test movement failed')
                return None

            result_pose = list(res.current_position)
            self.get_logger().info(
                '[TEST_MOVE_ABOVE result]\n'
                f'X={result_pose[0]:.2f}\n'
                f'Y={result_pose[1]:.2f}\n'
                f'Z={result_pose[2]:.2f}\n'
                f'Rx={result_pose[3]:.2f}\n'
                f'Ry={result_pose[4]:.2f}\n'
                f'Rz={result_pose[5]:.2f}\n'
                f'arm_state={res.arm_state}')
            if res.arm_state != RobotCommand.Response.IDLE:
                self.get_logger().error(
                    f'Robot did not reach test position, arm_state={res.arm_state}')
                return None

            with self._lock:
                self.current_pose = list(res.current_position)

            self.get_logger().info(
                'Position test completed. Robot is stopped above the selected ball. '
                'No second calibration, lowering, or hit was performed. '
                'Press Reset / Watch to return to the photo pose.')

            # Hold this position until the UI explicitly requests Reset.
            self._reset_event.clear()
            while rclpy.ok():
                if self._reset_event.wait(timeout=0.1):
                    self.get_logger().info(
                        'Position test reset requested: returning to photo pose')
                    return States.MOVE_TO_PHOTO
            return States.FINISH

        elif state == States.STRATEGY:
            self._light(False)
            actual_x, actual_y = [], []
            for i in range(0, len(self.ball_pose_buffer), 2):
                rel  = pixel_mm_convert(CAM_TO_TABLE, self.ball_pose_buffer[i:i+2])
                pose = convert_arm_pose(rel, FIX_ABS_CAM)
                actual_x.append(pose[0]); actual_y.append(pose[1])
            if len(self.updated_target_cue) >= 4:
                actual_x[0]  = self.updated_target_cue[0]
                actual_y[0]  = self.updated_target_cue[1]
                actual_x[-1] = self.updated_target_cue[-2]
                actual_y[-1] = self.updated_target_cue[-1]
            cuex = self.updated_target_cue[-2]
            cuey = self.updated_target_cue[-1]
            self.strategy_info = table2.main(actual_x[:-1], actual_y[:-1], cuex, cuey)
            return States.HITPOINT_PITCH

        elif state == States.HITPOINT_PITCH:
            self.obstacle  = self.strategy_info[3]
            self.hitpointx = self.strategy_info[4]
            self.hitpointy = self.strategy_info[5]
            self.score     = self.strategy_info[0]
            res = self._call(self._req(cmd_mode=RobotCommand.Request.CHECK_POSE,
                                       tool=CUE_TOOL))
            if res is None:
                return None
            self.current_tool_pose = list(res.current_position)
            pose = self._twist(PITCH_X, PITCH_Y, PITCH_Z,
                               self.current_tool_pose[3],
                               PITCH_ANGLE_Y if self.obstacle == 1
                               else self.current_tool_pose[4],
                               self.current_tool_pose[5])
            if self.obstacle != 1:
                pose.angular.x = self.current_tool_pose[3]
                pose.angular.y = self.current_tool_pose[4]
                pose.angular.z = self.current_tool_pose[5]
            self._call(self._req(cmd_mode=RobotCommand.Request.PTP,
                                 tool=CUE_TOOL, pose=pose))
            return States.CHECK_POSE

        elif state == States.CHECK_POSE:
            res = self._call(self._req(cmd_mode=RobotCommand.Request.CHECK_POSE))
            if res is None:
                return None
            with self._lock:
                self.current_pose = list(res.current_position)
            return States.HITPOINT_ANGLE

        elif state == States.HITPOINT_ANGLE:
            vx, vy  = self.strategy_info[1], self.strategy_info[2]
            yaw, _  = yaw_angle(vx, vy)
            pose = self._twist(*self.current_pose[:3],
                               self.current_pose[3],
                               self.current_pose[4],
                               yaw - 90.)
            self._call(self._req(cmd_mode=RobotCommand.Request.PTP,
                                 tool=CUE_TOOL, pose=pose))
            return States.HITPOINT_TOP

        elif state == States.HITPOINT_TOP:
            res = self._call(self._req(cmd_mode=RobotCommand.Request.CHECK_POSE,
                                       tool=CUE_TOOL))
            if res is None:
                return None
            self.current_tool_pose = list(res.current_position)
            pose = self._twist(self.hitpointx, self.hitpointy, HIT_TOP_Z,
                               *self.current_tool_pose[3:])
            self._call(self._req(cmd_mode=RobotCommand.Request.PTP,
                                 tool=CUE_TOOL, pose=pose))
            return States.HITBALL_POSE

        elif state == States.HITBALL_POSE:
            res = self._call(self._req(cmd_mode=RobotCommand.Request.CHECK_POSE,
                                       tool=CUE_TOOL))
            if res is None:
                return None
            self.current_pose = list(res.current_position)
            z = HIT_FLAT_Z if self.obstacle == 0 else HIT_OBSTACLE_Z
            pose = self._twist(self.hitpointx, self.hitpointy, z,
                               *self.current_pose[3:])
            res = self._call(self._req(cmd_mode=RobotCommand.Request.PTP,
                                       tool=CUE_TOOL, pose=pose))
            if res is None or res.arm_state != RobotCommand.Response.IDLE:
                return None
            return States.HITBALL

        elif state == States.HITBALL:
            score = self.strategy_info[0]
            hitpin = (HITHEAVY_PIN if score <= 3500
                      else HITMID_PIN if score <= 5000
                      else HITSOFT_PIN)
            self._call(self._req(cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                                 digital_output_cmd=RobotCommand.Request.DIGITAL_ON,
                                 digital_output_pin=hitpin))
            pose = self._twist(self.hitpointx, self.hitpointy, HIT_TOP_Z,
                               *self.current_pose[3:])
            self._call(self._req(cmd_mode=RobotCommand.Request.PTP,
                                 tool=CUE_TOOL, pose=pose))
            for pin in [hitpin, HEAVY_PIN]:
                self._call(self._req(cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                                     digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
                                     digital_output_pin=pin))
            return States.AF_HIT_TOP

        elif state == States.AF_HIT_TOP:
            pose = self._twist(self.hitpointx, self.hitpointy, HIT_TOP_Z,
                               -180., 0., 90.)
            self._call(self._req(cmd_mode=RobotCommand.Request.PTP,
                                 holding=False, tool=CUE_TOOL, pose=pose))
            # Reset for next round
            self._shot_event.clear()
            with self._lock:
                self._yolo_active = True
                self.ui_shot_info = None
            return States.INIT

        elif state == States.FINISH:
            return States.FINISH
        else:
            self.get_logger().error(f'Unknown state: {state}')
            return None

    def _main_loop(self):
        # INIT immediately forwards to MOVE_TO_PHOTO; no home-joint motion.
        state = States.INIT
        while state not in (States.FINISH, None) and rclpy.ok():
            try:
                state = self._sm(state)
            except Exception as e:
                self.get_logger().error(f'State machine error: {e}')
                state = None
        self.get_logger().info('Arm loop ended')

    def start(self):
        Thread(target=self._main_loop, daemon=True, name='arm_loop').start()

    # ── Service helpers ───────────────────────────
    @staticmethod
    def _twist(lx, ly, lz, ax=0., ay=0., az=0.):
        t = Twist()
        t.linear.x, t.linear.y, t.linear.z   = float(lx), float(ly), float(lz)
        t.angular.x, t.angular.y, t.angular.z = float(ax), float(ay), float(az)
        return t

    def _light(self, on):
        cmd = (RobotCommand.Request.DIGITAL_ON if on
               else RobotCommand.Request.DIGITAL_OFF)
        self._call(self._req(cmd_mode=RobotCommand.Request.DIGITAL_OUTPUT,
                             digital_output_cmd=cmd,
                             digital_output_pin=LIGHT_PIN))

    def _req(self, holding=True,
             cmd_mode=RobotCommand.Request.PTP,
             cmd_type=RobotCommand.Request.POSE_CMD,
             velocity=DEFAULT_VELOCITY, acceleration=DEFAULT_ACCEL,
             tool=1, base=0,
             digital_input_pin=0, digital_output_pin=0,
             digital_output_cmd=RobotCommand.Request.DIGITAL_OFF,
             pose=None, joints=None, circ_s=None, circ_end=None,
             jog_joint=6, jog_dir=0):
        if pose     is None: pose     = Twist()
        if joints   is None: joints   = [float('inf')]*6
        if circ_s   is None: circ_s   = []
        if circ_end is None: circ_end = []
        r = RobotCommand.Request()
        r.digital_input_pin  = digital_input_pin
        r.digital_output_pin = digital_output_pin
        r.digital_output_cmd = digital_output_cmd
        r.acceleration       = acceleration
        r.jog_joint          = jog_joint
        r.velocity           = velocity
        r.tool               = tool; r.base = base
        r.cmd_mode           = cmd_mode
        r.cmd_type           = cmd_type
        r.circ_end           = circ_end
        r.jog_dir            = jog_dir
        r.holding            = holding
        r.joints             = joints
        r.circ_s             = circ_s
        r.pose               = pose
        return r

    def _call(self, req):
        while not self.hiwin_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for hiwinmodbus_service...')
        future = self.hiwin_client.call_async(req)
        t0 = time.time()
        while not future.done():
            time.sleep(0.01)
            if time.time() - t0 > 30.0:
                self.get_logger().error('Service call timed out')
                return None
        return future.result()

    def _wait_future(self, future, timeout=30.0):
        t0 = time.time()
        while not future.done():
            time.sleep(0.01)
            if time.time()-t0 > timeout:
                return False
        return True


# ═══════════════════════════════════════════════════
#  Shot Planner UI
# ═══════════════════════════════════════════════════
class BilliardsUI:

    def __init__(self, root, node: Hiwin_Controller):
        self.root  = root
        self.node  = node
        root.title('Hiwin Billiards Shot Planner')
        root.configure(bg='#1a1a1a')
        root.resizable(False, False)

        # UI state
        self.ui_state      = UI_WATCH
        self.frozen_balls  = []
        self.white_ball    = None
        self.target_ball   = None
        self.target_pocket = None
        self.best_angle    = 0.0
        self.ghost_pos     = None
        self._drag_white   = False
        self._watching     = True
        self._manual_mode  = False   # False=auto-angle, True=manual drag angle

        self._build_ui()
        self._refresh()

    # ── Build UI ──────────────────────────────────
    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg='#111', height=36)
        top.pack(fill='x')
        self.lbl_state = tk.Label(top, text='WATCHING', bg='#111',
            fg='#27ae60', font=('Helvetica', 11, 'bold'), padx=12)
        self.lbl_state.pack(side='left', pady=6)
        self.lbl_hint = tk.Label(top, text='Waiting for YOLO...',
            bg='#111', fg='#888', font=('Helvetica', 10), padx=8)
        self.lbl_hint.pack(side='left', pady=6)

        # Mode toggle (top right)
        self.mode_var = tk.BooleanVar(value=False)
        self.btn_mode = tk.Checkbutton(top,
            text='Manual angle', variable=self.mode_var,
            bg='#111', fg='#aaa', selectcolor='#333',
            activebackground='#111', font=('Helvetica', 10),
            command=self._on_mode_change)
        self.btn_mode.pack(side='right', padx=12, pady=6)

        main = tk.Frame(self.root, bg='#1a1a1a')
        main.pack(fill='both', expand=True, padx=8, pady=8)

        # Canvas
        self.canvas = tk.Canvas(main, width=UW, height=UH,
            bg='#155c2c', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, rowspan=30, padx=(0, 10))
        self.canvas.bind('<ButtonPress-1>',   self._on_click)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        # Right panel
        rp = tk.Frame(main, bg='#1a1a1a', width=210)
        rp.grid(row=0, column=1, sticky='nw')
        rp.grid_columnconfigure(0, weight=1)

        def sec(t, r):
            tk.Label(rp, text=t, bg='#1a1a1a', fg='#555',
                     font=('Helvetica', 9)).grid(
                         row=r, column=0, columnspan=2, sticky='w', pady=(10, 2))

        def sldr(r, var, lo, hi, val):
            s = tk.Scale(rp, variable=var, from_=lo, to=hi, orient='horizontal',
                         length=190, bg='#222', fg='#eee', troughcolor='#444',
                         highlightthickness=0,
                         command=lambda _: self._on_param_change())
            s.set(val)
            s.grid(row=r, column=0, columnspan=2, pady=2)
            return s

        sec('FORCE', 0)
        self.v_force = tk.IntVar(value=6)
        sldr(1, self.v_force, 1, 10, 6)

        sec('SPIN', 2)
        self.v_spin = tk.IntVar(value=0)
        sldr(3, self.v_spin, -5, 5, 0)

        sec('MANUAL ANGLE (deg)', 4)
        self.v_angle = tk.IntVar(value=0)
        self.sldr_angle = sldr(5, self.v_angle, 0, 359, 0)
        self.sldr_angle.config(state='disabled')  # enabled in manual mode

        self.v_bounce = tk.BooleanVar(value=True)
        tk.Checkbutton(rp, text='Show bounces', variable=self.v_bounce,
            bg='#1a1a1a', fg='#aaa', selectcolor='#333',
            activebackground='#1a1a1a',
            command=self._draw).grid(row=6, column=0, columnspan=2,
                                     sticky='w', pady=4)

        ttk.Separator(rp).grid(row=7, column=0, columnspan=2,
                               sticky='ew', pady=6)
        sec('TARGET POCKET', 8)
        self.pocket_var = tk.StringVar(value='')
        for i, (px, py, name, _) in enumerate(POCKET_POSITIONS):
            tk.Radiobutton(rp, text=name, variable=self.pocket_var, value=name,
                bg='#1a1a1a', fg='#aaa', selectcolor='#27ae60',
                activebackground='#1a1a1a',
                command=self._on_pocket_select).grid(
                    row=9 + i//3, column=i % 3, sticky='w', padx=4)

        ttk.Separator(rp).grid(row=11, column=0, columnspan=2,
                               sticky='ew', pady=6)
        sec('SHOT INFO', 12)
        self.lbl_angle  = self._info_row(rp, 'Angle',  13)
        self.lbl_force2 = self._info_row(rp, 'Force',  14)
        self.lbl_spin2  = self._info_row(rp, 'Spin',   15)
        self.lbl_pred   = self._info_row(rp, 'Pocket', 16)
        self.lbl_wb     = self._info_row(rp, 'W-ball', 17)

        ttk.Separator(rp).grid(row=18, column=0, columnspan=2,
                               sticky='ew', pady=6)

        def big_btn(text, color, row, cmd, state='normal'):
            b = tk.Button(rp, text=text, bg=color, fg='#fff',
                          font=('Helvetica', 10, 'bold'), relief='flat',
                          padx=6, pady=6, command=cmd, state=state)
            b.grid(row=row, column=0, columnspan=2, sticky='ew', pady=2)
            return b

        self.btn_freeze = big_btn('Freeze Balls',      '#f39c12', 19, self._freeze_balls)
        self.btn_auto   = big_btn('Auto Best Shot',    '#2980b9', 20, self._auto_calculate)
        self.btn_send   = big_btn('MOVE ABOVE BALL ▶', '#27ae60', 21, self._send_shot,
                                  state='disabled')
        self.btn_reset  = big_btn('Reset / Watch',     '#555555', 22, self._reset)

        self.status = tk.Label(self.root, text='Waiting for YOLO...', anchor='w',
            bg='#111', fg='#888', font=('Courier', 9), padx=8)
        self.status.pack(fill='x', side='bottom')

        self.lbl_pose = tk.Label(self.root, text='Arm pos: --', anchor='w',
            bg='#0a0a0a', fg='#27ae60', font=('Courier', 9), padx=8)
        self.lbl_pose.pack(fill='x', side='bottom')

    def _info_row(self, parent, label, row):
        tk.Label(parent, text=label+':', bg='#1a1a1a', fg='#555',
                 font=('Helvetica', 9)).grid(row=row, column=0, sticky='w', padx=4)
        lbl = tk.Label(parent, text='--', bg='#1a1a1a', fg='#eee',
                       font=('Helvetica', 9, 'bold'))
        lbl.grid(row=row, column=1, sticky='w', padx=4)
        return lbl

    # ── Mode switch ───────────────────────────────
    def _on_mode_change(self):
        self._manual_mode = self.mode_var.get()
        if self._manual_mode:
            self.sldr_angle.config(state='normal')
            self._set_state('MANUAL MODE', 'Set angle with slider, click ball', '#e67e22')
        else:
            self.sldr_angle.config(state='disabled')
            self._set_state('AUTO MODE', 'Click target ball then pocket', '#2980b9')
        if self.ui_state == UI_PREVIEW:
            self._calculate_shot()
        self._draw()

    # ── Ball freeze ───────────────────────────────
    def _freeze_balls(self):
        live = self.node.get_balls_for_ui()
        if not live:
            messagebox.showwarning('No balls', 'No YOLO detections yet.')
            return
        self.frozen_balls = live
        self._watching = False
        self.white_ball = None
        for b in self.frozen_balls:
            if b['label'].lower() in ('white', 'cue', 'w'):
                self.white_ball = dict(b); break
        if not self.white_ball:
            self.white_ball = {'x': UW//2-80, 'y': UH//2, 'label': 'W'}
        self.ui_state = UI_SELECT
        self._set_state('SELECT BALL', 'Click any ball to test its position', '#f39c12')
        self._draw()
        self.status['text'] = f'Frozen {len(live)} balls'

    # ── Pocket selection ──────────────────────────
    def _on_pocket_select(self):
        name = self.pocket_var.get()
        for px, py, n, _ in POCKET_POSITIONS:
            if n == name:
                self.target_pocket = (px, py, n); break
        if self.target_ball:
            self._calculate_shot()
        self._draw()

    def _on_param_change(self):
        if self.ui_state == UI_PREVIEW:
            self._calculate_shot()
        self._draw()

    # ── Shot calculation ──────────────────────────
    def _calculate_shot(self):
        if not (self.white_ball and self.target_ball and self.target_pocket):
            return
        wx, wy = self.white_ball['x'], self.white_ball['y']
        tx, ty = self.target_ball['x'], self.target_ball['y']
        px, py, _ = self.target_pocket
        force  = self.v_force.get()
        spin   = self.v_spin.get()
        bounce = self.v_bounce.get()

        if self._manual_mode:
            angle = float(self.v_angle.get())
            rad   = math.radians(angle)
            # ghost: white ball travels along manual angle
            dist  = math.hypot(tx-wx, ty-wy)
            gx    = wx + math.cos(rad)*dist
            gy    = wy + math.sin(rad)*dist
        else:
            angle, gx, gy = ui_calc_hit_angle(wx, wy, tx, ty, px, py)
            self.v_angle.set(int(angle) % 360)

        self.best_angle = angle
        self.ghost_pos  = (gx, gy)

        rad   = math.radians(angle)
        speed = force * 3.5
        wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
        dx, dy   = gx-wx, gy-wy
        dist     = max(1, math.hypot(dx, dy))
        nx_, ny_ = dx/dist, dy/dist
        spin_eff = spin * 0.35
        (tvx, tvy), _ = ui_collision(wvx, wvy, gx, gy, tx, ty)
        tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
        tp = ui_simulate_path(tx, ty, tvx, tvy, force*50, bounce)
        last  = tp[-1]
        best_p = min(POCKET_POSITIONS,
                     key=lambda p: math.hypot(p[0]-last[0], p[1]-last[1]))
        best_d  = math.hypot(best_p[0]-last[0], best_p[1]-last[1])

        self.lbl_angle['text']  = f'{angle:.1f}°'
        self.lbl_force2['text'] = str(force)
        self.lbl_spin2['text']  = str(spin)
        self.lbl_pred['text']   = best_p[2] if best_d < 60 else 'Miss'
        self.lbl_wb['text']     = f'{wx:.0f},{wy:.0f}px'

        self.ui_state = UI_PREVIEW
        self._set_state('PREVIEW', 'Review then click MOVE ABOVE BALL', '#2980b9')
        self.btn_send['state'] = 'normal'

    def _auto_calculate(self):
        if not self.frozen_balls:
            messagebox.showinfo('Freeze first', 'Press Freeze Balls first.'); return
        if not self.target_pocket:
            messagebox.showinfo('Select pocket', 'Select a pocket first.'); return
        balls = [b for b in self.frozen_balls
                 if b['label'].lower() not in ('white', 'cue', 'w')]
        if not balls:
            messagebox.showinfo('No balls', 'No target balls detected.'); return
        px, py, _ = self.target_pocket
        wx = self.white_ball['x'] if self.white_ball else UW//2
        wy = self.white_ball['y'] if self.white_ball else UH//2
        force = self.v_force.get(); spin = self.v_spin.get()
        best, best_d = None, 9999
        for b in balls:
            angle, gx, gy = ui_calc_hit_angle(wx, wy, b['x'], b['y'], px, py)
            rad = math.radians(angle); speed = force*3.5
            wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
            dx, dy = gx-wx, gy-wy; dist = max(1, math.hypot(dx, dy))
            nx_, ny_ = dx/dist, dy/dist; spin_eff = spin*0.35
            (tvx, tvy), _ = ui_collision(wvx, wvy, gx, gy, b['x'], b['y'])
            tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
            tp = ui_simulate_path(b['x'], b['y'], tvx, tvy, force*50, True)
            d  = math.hypot(tp[-1][0]-px, tp[-1][1]-py)
            if d < best_d:
                best_d = d; best = b
        self.target_ball = dict(best)
        self._calculate_shot()
        self._draw()
        self.status['text'] = (f'Auto: Ball {best["label"]} '
                               f'angle={self.best_angle:.1f}°')

    # ── Send shot ─────────────────────────────────
    def _send_shot(self):
        """Send only the selected target-ball position for an X/Y accuracy test."""
        if self.target_ball is None:
            messagebox.showerror('No ball selected', 'Freeze balls, then click a ball first.')
            return
        if not self.target_ball:
            messagebox.showerror('No target ball',
                                 'Please select a target ball first.')
            return

        # Use the Homography Base coordinates saved with the selected ball.
        arm_x = float(self.target_ball['arm_x'])
        arm_y = float(self.target_ball['arm_y'])

        white_yolo = None
        if (self.white_ball and 'yolo_x' in self.white_ball
                and 'yolo_y' in self.white_ball):
            white_yolo = [self.white_ball['yolo_x'],
                          self.white_ball['yolo_y']]
        target_yolo = [self.target_ball['yolo_x'],
                       self.target_ball['yolo_y']]

        self.node.get_logger().info(
            '[UI selected]\n'
            f'Ball={self.target_ball["label"]}\n'
            f'YOLO=({float(self.target_ball["yolo_x"]):.2f}, '
            f'{float(self.target_ball["yolo_y"]):.2f})\n'
            f'Homography Base=({arm_x:.2f}, {arm_y:.2f})\n'
            f'Arm command=({arm_x:.2f}, {arm_y:.2f}, '
            f'Z={TEST_ABOVE_Z:.2f})\n'
            f'Tool={CUE_TOOL}')

        self.node.set_shot_from_ui(
            arm_x, arm_y, self.best_angle, self.v_force.get(),
            white_yolo=white_yolo, target_yolo=target_yolo)

        self.ui_state = UI_HITTING
        self._set_state('POSITION TEST...',
                        'Robot moving above selected ball', '#e74c3c')
        self.btn_send['state'] = 'disabled'
        self.status['text'] = (
            f'Position test -> target arm X={arm_x:.1f}, Y={arm_y:.1f}, '
            f'Z={TEST_ABOVE_Z:.1f}')

        # Do not auto-reset. The robot must remain stopped above the ball
        # so the operator can measure the X/Y error.

    def _reset(self):
        self.node.request_photo_pose()
        self.ui_state = UI_WATCH
        self.frozen_balls  = []
        self.white_ball    = None
        self.target_ball   = None
        self.target_pocket = None
        self.ghost_pos     = None
        self._watching     = True
        self.pocket_var.set('')
        self.btn_send['state'] = 'disabled'
        for lbl in (self.lbl_angle, self.lbl_force2, self.lbl_spin2,
                    self.lbl_pred,  self.lbl_wb):
            lbl['text'] = '--'
        self._set_state('MOVING TO PHOTO...', 'Arm returning to camera position', '#e67e22')
        self._draw()

    def _set_state(self, s, h, c):
        self.lbl_state['text'] = s
        self.lbl_state['fg']   = c
        self.lbl_hint['text']  = h

    # ── Canvas interaction ────────────────────────
    def _on_click(self, event):
        mx, my = event.x, event.y
        if self.ui_state == UI_WATCH:
            return

        # Position-test mode: allow selecting ANY frozen ball, including white.
        if self.ui_state in (UI_SELECT, UI_PREVIEW):
            nearest = None
            nearest_d = float('inf')
            for b in self.frozen_balls:
                d = math.hypot(mx - b['x'], my - b['y'])
                if d < nearest_d:
                    nearest_d = d
                    nearest = b

            if nearest is not None and nearest_d <= BR + 10:
                self.target_ball = dict(nearest)
                self.ui_state = UI_PREVIEW
                self.btn_send['state'] = 'normal'
                self._set_state(
                    'BALL SELECTED',
                    'Press MOVE ABOVE SELECTED BALL',
                    '#2980b9'
                )
                self.status['text'] = (
                    f'Selected ball {nearest["label"]}: '
                    f'canvas=({nearest["x"]:.1f}, {nearest["y"]:.1f})'
                )
                self._draw()
                return

    def _on_drag(self, event):
        if self._drag_white and self.white_ball:
            self.white_ball['x'] = max(UPAD+BR+2, min(UW-UPAD-BR-2, event.x))
            self.white_ball['y'] = max(UPAD+BR+2, min(UH-UPAD-BR-2, event.y))
            if self.target_ball and self.target_pocket:
                self._calculate_shot()
            self._draw()

    def _on_release(self, _):
        self._drag_white = False

    # ── Drawing ───────────────────────────────────
    def _draw(self):
        c = self.canvas
        c.delete('all')
        # Table
        c.create_rectangle(UPAD//2, UPAD//2, UW-UPAD//2, UH-UPAD//2,
                            fill='#8B4513', outline='')
        c.create_rectangle(UPAD, UPAD, UW-UPAD, UH-UPAD,
                            fill='#155c2c', outline='#0d3d1c', width=1.5)
        c.create_line(UW//2, UPAD, UW//2, UH-UPAD, fill='#1a7a3a', dash=(4, 10))
        c.create_oval(UW//2-35, UH//2-35, UW//2+35, UH//2+35,
                      fill='', outline='#1a7a3a')
        # Pockets
        for px, py, name, _ in POCKET_POSITIONS:
            sel = self.target_pocket and self.target_pocket[2] == name
            c.create_oval(px-PR, py-PR, px+PR, py+PR,
                          fill='#ffff00' if sel else '#000',
                          outline='#ffcc00' if sel else '#333',
                          width=2 if sel else 1)
            c.create_text(px, py, text=name, fill='#ffcc00' if sel else '#555',
                          font=('Helvetica', 6, 'bold'))
        # Preview lines
        if self.ui_state == UI_PREVIEW and self.white_ball and self.target_ball:
            self._draw_preview(c)
        # Balls
        balls = (self.frozen_balls if not self._watching
                 else self.node.get_balls_for_ui())
        for b in balls:
            bx, by   = b['x'], b['y']
            label    = b['label']
            color    = BALL_COLORS.get(label.lower(), '#888888')
            is_white = label.lower() in ('white', 'cue', 'w')
            is_tgt   = (self.target_ball and
                        abs(bx-self.target_ball['x']) < 3 and
                        abs(by-self.target_ball['y']) < 3)
            if is_tgt:
                c.create_oval(bx-BR-6, by-BR-6, bx+BR+6, by+BR+6,
                              fill='', outline='#ff4444', width=2, dash=(3, 3))
            c.create_oval(bx-BR, by-BR, bx+BR, by+BR, fill=color,
                          outline='#cccccc' if is_white else '#333333', width=1.5)
            c.create_text(bx, by, text=label[:2],
                          fill='#333' if is_white else '#fff',
                          font=('Helvetica', 7, 'bold'))
        # White ball overlay
        if self.white_ball and not self._watching:
            wx, wy = self.white_ball['x'], self.white_ball['y']
            c.create_oval(wx-BR-2, wy-BR-2, wx+BR+2, wy+BR+2,
                          fill='#ffffff', outline='#aaaaaa', width=1.5)
            c.create_text(wx, wy,      text='W', fill='#888', font=('Helvetica', 7, 'bold'))
            c.create_text(wx, wy+BR+9, text='drag',fill='#888888',font=('Helvetica', 7))
        # Mode indicator
        mode_txt = '🔧 MANUAL' if self._manual_mode else '🤖 AUTO'
        c.create_text(UW-UPAD-4, UH-UPAD-8, anchor='se',
                      text=mode_txt, fill='#27ae60', font=('Helvetica', 8))
        count = len(balls)
        c.create_text(UPAD+4, UH-UPAD-8, anchor='sw',
                      text=f'{count} balls', fill='#1d7a3a', font=('Helvetica', 8))
        if self._watching and count == 0:
            c.create_text(UW//2, UH//2, text='Waiting for YOLO...',
                          fill='#1d7a3a', font=('Helvetica', 12))

    def _draw_preview(self, c):
        wx, wy = self.white_ball['x'], self.white_ball['y']
        tx, ty = self.target_ball['x'], self.target_ball['y']
        px, py, _ = self.target_pocket
        force  = self.v_force.get()
        spin   = self.v_spin.get()
        bounce = self.v_bounce.get()
        rad    = math.radians(self.best_angle)
        speed  = force * 3.5
        wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
        gx, gy   = self.ghost_pos if self.ghost_pos else (tx, ty)
        # Ghost ball
        c.create_oval(gx-BR, gy-BR, gx+BR, gy+BR,
                      fill='', outline='#888888', dash=(4, 4))
        c.create_line(wx, wy, gx, gy, fill='#555555', dash=(4, 8))
        c.create_line(tx, ty, px, py, fill='#883333', dash=(3, 6))
        dx, dy = gx-wx, gy-wy
        dist   = max(1, math.hypot(dx, dy))
        nx_, ny_ = dx/dist, dy/dist
        spin_eff = spin*0.35
        steps1 = max(1, int(dist/speed))
        wp = ui_simulate_path(wx, wy, wvx, wvy, steps1, False)
        self._path(c, wp, '#ffffff', 2, (6, 4))
        (tvx, tvy), (rvx, rvy) = ui_collision(wvx, wvy, gx, gy, tx, ty)
        tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
        tp = ui_simulate_path(tx, ty, tvx, tvy, force*50, bounce)
        self._path(c, tp, '#ff5555', 2.5)
        if len(tp) > 1:
            self._arrow(c, tp[-2], tp[-1], '#ff3333')
        wp2 = ui_simulate_path(gx, gy, rvx, rvy, force*30, bounce)
        self._path(c, wp2, '#ffff88', 1, (3, 6))
        ax = wx + math.cos(rad)*35
        ay = wy + math.sin(rad)*35
        c.create_line(wx, wy, ax, ay, fill='#ffffff', width=2.5)
        self._arrow(c, (wx, wy), (ax, ay), '#ffffff')

    def _path(self, c, path, color, width=1.5, dash=None):
        if len(path) < 2: return
        flat = [v for pt in path for v in pt]
        opts = dict(fill=color, width=width, smooth=True)
        if dash: opts['dash'] = dash
        c.create_line(*flat, **opts)

    def _arrow(self, c, tail, tip, color):
        dx, dy = tip[0]-tail[0], tip[1]-tail[1]
        mag = math.hypot(dx, dy)
        if mag == 0: return
        nx, ny = dx/mag, dy/mag
        ax, ay = tip
        c.create_polygon(ax, ay,
                         ax-nx*10+ny*4, ay-ny*10-nx*4,
                         ax-nx*10-ny*4, ay-ny*10+nx*4,
                         fill=color, outline='')

    def _refresh(self):
        if self._watching:
            self._draw()
        p = self.node.get_current_pose()
        if any(v != 0 for v in p):
            self.lbl_pose['text'] = (
                f'Arm pos:  X={p[0]:8.2f}  Y={p[1]:8.2f}  Z={p[2]:8.2f}'
                f'  Rx={p[3]:7.2f}  Ry={p[4]:7.2f}  Rz={p[5]:7.2f}  mm/deg')
        self.root.after(150, self._refresh)


# ═══════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = Hiwin_Controller()

    # Start ROS spinning first so service responses and YOLO callbacks work
    # while the arm is moving to the photo pose.
    ros_thread = Thread(target=lambda: rclpy.spin(node), daemon=True,
                        name='ros_spin')
    ros_thread.start()
    node.start()

    # Do not create Tkinter until arm.yaml -> armpos has been reached.
    while rclpy.ok():
        if node.photo_pose_ready.wait(timeout=0.1):
            break
        if node.startup_failed.is_set():
            node.get_logger().error('Startup stopped because photo-pose movement failed')
            node.destroy_node()
            rclpy.shutdown()
            return

    if not rclpy.ok():
        return

    # Tkinter must run in the main thread. It opens only after the move completes.
    root = tk.Tk()
    app  = BilliardsUI(root, node)

    def _shutdown(*_):
        """Called on SIGINT / SIGTERM — tears down ROS then closes the window."""
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        try:
            root.destroy()
        except Exception:
            pass

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    def _poll_ros():
        """Periodic check: close UI if ROS has already shut down."""
        if not rclpy.ok():
            try:
                root.destroy()
            except Exception:
                pass
            return
        root.after(500, _poll_ros)

    root.after(500, _poll_ros)

    try:
        root.mainloop()
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        os.system('clear')


if __name__ == '__main__':
    main()
