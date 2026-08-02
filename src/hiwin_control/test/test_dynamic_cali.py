"""DYNAMIC_CALI 視覺伺服迴圈的離線測試（不需要手臂、不需要相機、不需要球桌）。

測的是 `arm_controler_track.py`（不是 arm_controler_how.py）。

兩個 seam：
  1. `Hiwin_Controller.call_hiwin` -> 假手臂。收到 PTP 就記住新位置，
     收到 CHECK_POSE 就把記住的位置回報出去。真手臂會做的事只有這兩件。
  2. `node.all_ball_pose` -> 假相機。手臂每動一次就重算一次「球現在落在哪個像素」，
     等同於 YOLO 送來一幀新影像。

假相機刻意「從針孔投影的第一原理」獨立寫一次，不拿 pixel_mm_convert 反推當標準答案，
否則不管被測程式對錯測試都會過（自己考自己）。

在 container 內從 workspace 根目錄執行：
    cd /home/how/work && python3 -m pytest src/hiwin_control/test/test_dynamic_cali.py -v
"""
import math
import os
import sys

import numpy as np
import pytest
from hiwin_interfaces.srv import RobotCommand

_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_PKG_SRC = os.path.join(_WS_ROOT, 'src', 'hiwin_control')
sys.path.insert(0, _PKG_SRC)

# 模組在 import 當下就用 os.getcwd() 去讀 .ini / arm.yaml，先切到 workspace 根目錄。
_PREV_CWD = os.getcwd()
os.chdir(_WS_ROOT)
try:
    import hiwin_control.arm_controler_track as arm
    from hiwin_control import transformations
finally:
    os.chdir(_PREV_CWD)

assert os.path.abspath(arm.__file__).startswith(_PKG_SRC), (
    '測試匯入到錯的程式碼：' + arm.__file__ + '，預期在 ' + _PKG_SRC)


# --------------------------------------------------------------------------
# 假相機：世界座標 -> 像素
# --------------------------------------------------------------------------
# 手眼標定的平移/旋轉。跟 arm_controler_track.convert_arm_pose() 裡的硬寫值同源，
# 但這裡只用來「造出影像」，被測程式是從影像反推座標 —— 方向相反，不構成循環論證。
TOOL2CAM_Q = [0.003109262802720864, -0.002553016790632168,
              0.7071215509442967, 0.7070805659754924]      # qx qy qz qw
TOOL2CAM_T = [0.11155349816142887, 0.0022825871424820617,
              -0.06817001513263532]                        # x y z (m)

# 完全垂直的理想手眼：拿掉光軸傾斜，用來隔離「迴圈邏輯」與「標定誤差」。
# qz=qw=1/sqrt(2) 是繞 Z 轉 90 度；tool2cam 的 x/y 對調就是從這裡來的。
IDEAL_Q = [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
IDEAL_T = [TOOL2CAM_T[0], TOOL2CAM_T[1], TOOL2CAM_T[2]]

TABLE_Z = arm.FIX_ABS_CAM[2] + arm.TOOL_TO_CAM[2] - arm.CAM_TO_TABLE
BALL_Z = TABLE_Z + arm.BALL_R


def _quat_to_mat(qx, qy, qz, qw):
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def _homo(rot, trans_m):
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = trans_m
    return m


def _base_to_cam(arm_pose, quat, trans):
    d2r = math.pi / 180.0
    q = transformations.quaternion_from_euler(
        arm_pose[3] * d2r, arm_pose[4] * d2r, arm_pose[5] * d2r, axes='sxyz')
    base2tool = _homo(_quat_to_mat(q[0], q[1], q[2], q[3]), np.array(arm_pose[:3]) / 1000.0)
    return base2tool @ _homo(_quat_to_mat(*quat), np.array(trans))


def project(ball_world_mm, arm_pose, quat=TOOL2CAM_Q, trans=TOOL2CAM_T):
    """針孔投影：球的世界座標(mm) -> 像素 (u, v)。球在相機後方時回傳 None。"""
    b2c = _base_to_cam(arm_pose, quat, trans)
    p_cam = np.linalg.inv(b2c) @ np.array([ball_world_mm[0] / 1000.0,
                                           ball_world_mm[1] / 1000.0,
                                           ball_world_mm[2] / 1000.0, 1.0])
    if p_cam[2] <= 0:
        return None
    return (arm.UNDIST_CX + arm.UNDIST_FX * p_cam[0] / p_cam[2],
            arm.UNDIST_CY + arm.UNDIST_FY * p_cam[1] / p_cam[2])


IMG_W, IMG_H = 1920, 1080


def tilt_offset_mm(fix_z, quat=TOOL2CAM_Q, trans=TOOL2CAM_T):
    """光軸傾斜在給定工作距離下造成的固定橫向偏移 = d * tan(傾斜角)。"""
    axis = _base_to_cam(arm.FIX_ABS_CAM, quat, trans)[:3, 2]
    tilt = math.acos(float(np.dot(axis, [0, 0, -1])))
    d = arm.cam_to_ball_height(fix_z, arm.TOOL_TO_CAM[2], TABLE_Z)
    return d * math.tan(tilt)


# --------------------------------------------------------------------------
# 假手臂 + 假相機的組合
# --------------------------------------------------------------------------
class FakeRig:
    """把假手臂跟假相機綁在一起：手臂一動，畫面就跟著更新。"""

    def __init__(self, node, balls_world, quat=TOOL2CAM_Q, trans=TOOL2CAM_T):
        self.node = node
        self.balls = list(balls_world)
        self.quat = quat
        self.trans = trans
        self.pose = list(arm.FIX_ABS_CAM)
        self.ptp_count = 0
        self._refresh_image()

    def _refresh_image(self):
        """把目前手臂姿態下看得到的球寫進 node.all_ball_pose（等同 YOLO 送來一幀）。"""
        pixels = []
        for ball in self.balls:
            uv = project(ball, self.pose, self.quat, self.trans)
            if uv is None:
                continue
            if 0 <= uv[0] < IMG_W and 0 <= uv[1] < IMG_H:
                pixels.extend([uv[0], uv[1]])
        self.node.all_ball_pose = pixels
        self.node.data_flag = 1 if pixels else 0

    def call_hiwin(self, req):
        res = RobotCommand.Response()
        res.arm_state = RobotCommand.Response.IDLE
        res.digital_state = 1
        if req.cmd_mode == RobotCommand.Request.PTP:
            self.ptp_count += 1
            self.pose = [req.pose.linear.x, req.pose.linear.y, req.pose.linear.z,
                         req.pose.angular.x, req.pose.angular.y, req.pose.angular.z]
            self._refresh_image()
        res.current_position = list(self.pose)
        return res


@pytest.fixture
def node():
    import rclpy
    rclpy.init()
    controller = arm.Hiwin_Controller()
    yield controller
    controller.destroy_node()
    rclpy.shutdown()


def run_dynamic_cali(node, balls_world, rough_positions, quat=TOOL2CAM_Q, trans=TOOL2CAM_T):
    """把狀態機直接放在 DYNAMIC_CALI 起跑，跑到它離開為止。回傳 (rig, 下一個 state)。

    LOCK_INFO 之後的節點狀態手動擺好 —— 這份測試只針對伺服迴圈，
    不重測第一張照片的換算。
    """
    rig = FakeRig(node, balls_world, quat, trans)
    node.call_hiwin = rig.call_hiwin
    node.ball_pose = [list(p) for p in rough_positions]
    node.target_cue = [[0.0, 0.0] for _ in balls_world]   # 只有 len() 會被用到
    node.updated_target_cue = []
    node.index = 0

    state = arm.States.DYNAMIC_CALI
    for _ in range(len(balls_world) + 2):
        state = node._state_machine(state)
        if state != arm.States.DYNAMIC_CALI:
            break
    return rig, state


# --------------------------------------------------------------------------
# 測試
# --------------------------------------------------------------------------
def test_recovers_known_ball_position_with_ideal_hand_eye(node):
    """理想手眼（光軸完全垂直）下，伺服迴圈必須把球的座標還原到 0.5mm 內。

    這是第 1 塊（target_cue 假偏移）的把關測試：那個 bug 沒修的話，
    最終座標會被硬加上「第一張照片像素」換算出的數十~數百 mm 偏移。
    """
    balls = [(150.0, 420.0, BALL_Z), (-220.0, 560.0, BALL_Z)]
    # 第一張照片的粗略位置故意各差 15mm —— 伺服迴圈本來就該把這個誤差吃掉。
    rough = [(165.0, 405.0), (-235.0, 575.0)]

    rig, state = run_dynamic_cali(node, balls, rough, quat=IDEAL_Q, trans=IDEAL_T)

    assert state == arm.States.STRATEGY
    assert len(node.updated_target_cue) == 4, '每顆球應該產生 2 筆座標'
    for i, ball in enumerate(balls):
        gx, gy = node.updated_target_cue[2 * i], node.updated_target_cue[2 * i + 1]
        err = math.hypot(gx - ball[0], gy - ball[1])
        assert err < 0.5, (
            'index_{} 還原誤差 {:.2f}mm：期望 ({:.1f},{:.1f})，得到 ({:.2f},{:.2f})'
            .format(i, err, ball[0], ball[1], gx, gy))


def test_converges_in_a_few_iterations(node):
    """Kp=0.25 每圈收掉 25% 誤差，粗略位置差 21mm 時應該 11 圈左右收斂。

    這條是深度公式的間接把關：等效增益 = Kp * (程式假設的深度 / 真實深度)。
    修好前的深度是真值的 0.65 倍 -> 等效增益 0.163 -> 需要約 18 圈，會超過上限。
    但它只驗得到「增益的量級」，驗不到最終座標 —— 收斂時殘差趨近 0，
    乘上任何深度都還是 0。
    """
    balls = [(150.0, 420.0, BALL_Z)]
    rough = [(165.0, 405.0)]

    rig, state = run_dynamic_cali(node, balls, rough, quat=IDEAL_Q, trans=IDEAL_T)

    assert state == arm.States.STRATEGY
    # 1 次移到校正位 + 迴圈內的修正次數。理論值 12，留一點餘裕；
    # 深度錯 1.5 倍以上就會超過。
    assert rig.ptp_count <= 14, '收斂用掉 {} 次 PTP，太多'.format(rig.ptp_count)


def test_falls_back_to_first_photo_when_ball_never_visible(node):
    """球不在畫面裡時，迴圈必須在 MAX_CALI_ITER 次後放棄並退回粗略座標，
    而不是無限迴圈把狀態機卡死（第 3a 塊）。
    """
    ball_world = (150.0, 420.0, BALL_Z)
    rough = [(165.0, 405.0)]
    # 桌上真正的球放在很遠的地方，相機開到 rough 位置時畫面裡只有它、且離中心極遠，
    # 迴圈會一直追而收斂不了。
    balls = [(150.0 + 900.0, 420.0, BALL_Z)]

    rig, state = run_dynamic_cali(node, balls, rough, quat=IDEAL_Q, trans=IDEAL_T)

    assert state == arm.States.STRATEGY, '沒收斂也必須離開 DYNAMIC_CALI'
    assert len(node.updated_target_cue) == 2
    assert rig.ptp_count <= arm.MAX_CALI_ITER + 2, (
        'PTP 次數 {} 超過上限，迴圈沒被 MAX_CALI_ITER 擋住'.format(rig.ptp_count))
    assert ball_world is not None


def test_pixel_to_world_xy_is_exact_across_the_whole_table():
    """射線求交必須在整張桌子(含最角落)都準到 0.05mm 內。

    對照:原本的「固定深度反投影 + convert_arm_pose」在同樣這些點是 1.6~4.0mm。
    那 1.6~4.0mm 是兩個誤差部分抵消後的殘值 ——
      (1) pixel_mm_convert 回傳的深度寫死 1.0 公尺(真值 0.676)-> 固定偏 2.61mm
      (2) dev 的深度也用常數,但光軸歪 0.4619 度使各處真實深度不同 -> 桌角約 3.6mm
    方向相反,所以只修其中一個會讓桌角變差。射線求交不假設深度,兩個一起消失。
    """
    corners = [(0.0, 400.0), (200.0, 400.0), (500.0, 700.0), (550.0, 730.0),
               (-550.0, 200.0), (-400.0, 300.0), (0.0, 730.0), (-560.0, 740.0)]
    for bx, by in corners:
        uv = project((bx, by, BALL_Z), arm.FIX_ABS_CAM)
        assert uv is not None
        xy = arm.pixel_to_world_xy(uv, arm.FIX_ABS_CAM)
        assert xy is not None
        err = math.hypot(xy[0] - bx, xy[1] - by)
        assert err < 0.05, (
            '球 ({:.1f},{:.1f}) 還原誤差 {:.4f}mm，得到 ({:.3f},{:.3f})'
            .format(bx, by, err, xy[0], xy[1]))


def test_pixel_to_world_xy_rejects_rays_that_miss_the_plane():
    """相機朝上(射線永遠打不到球心平面)時必須回 None，不能給出假座標。"""
    upside_down = [0.0, 334.047, 423.503, 0.0, 0.0, 90.0]   # Rx=0 -> 鏡頭朝上
    assert arm.pixel_to_world_xy([960.0, 540.0], upside_down) is None


def test_real_hand_eye_leaves_a_constant_tilt_offset(node):
    """用真實的 ini 手眼（光軸歪 0.46 度）時，殘留的是一個「與位置無關的固定偏移」，
    大小 = 工作距離 * tan(傾斜角)。

    這條測試把這個已知的系統性誤差釘住：它應該是常數（可以被實測式標定吸收掉），
    而不是隨球的位置變化（那樣就吸收不掉了）。
    """
    balls = [(150.0, 420.0, BALL_Z), (-300.0, 600.0, BALL_Z)]
    rough = [(160.0, 410.0), (-310.0, 610.0)]

    rig, state = run_dynamic_cali(node, balls, rough)

    assert state == arm.States.STRATEGY
    errors = []
    for i, ball in enumerate(balls):
        gx, gy = node.updated_target_cue[2 * i], node.updated_target_cue[2 * i + 1]
        errors.append((gx - ball[0], gy - ball[1]))

    expected = tilt_offset_mm(node.fix_z)
    for ex, ey in errors:
        assert abs(math.hypot(ex, ey) - expected) < 0.3, (
            '偏移量 {:.2f}mm 與預期的 d*tan(tilt)={:.2f}mm 不符'
            .format(math.hypot(ex, ey), expected))
    # 兩顆球的偏移方向與大小必須一致 —— 是常數才吸收得掉。
    assert abs(errors[0][0] - errors[1][0]) < 0.1
    assert abs(errors[0][1] - errors[1][1]) < 0.1
