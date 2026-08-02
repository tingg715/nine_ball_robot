"""狀態機的行為測試（不需要手臂、不需要球桌）。

Seam：`Hiwin_Controller.call_hiwin` —— 通往 modbus server / 真手臂的系統邊界。
測試在這條線上塞一個假的 response，然後觀察 `_state_machine` 的兩個對外行為：
下一個 state 是什麼、以及送出了哪些 request。

在 container 內從 workspace 根目錄執行：
    cd /home/how/work && python3 -m pytest src/hiwin_control/test/test_state_machine.py -v
"""
import os
import sys

import pytest
import rclpy
from hiwin_interfaces.srv import RobotCommand

# 測的必須是 src/ 底下的工作目錄程式碼，不是 install/ 那份 build 產物 ——
# container 裡 source 過 install/setup.bash，不指定的話 import 會解析到安裝目錄，
# 於是你改了 src 卻測到舊程式碼。把 src 插到 sys.path 最前面。
_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_PKG_SRC = os.path.join(_WS_ROOT, 'src', 'hiwin_control')
sys.path.insert(0, _PKG_SRC)

# arm_controler_how 在 import 當下就用 os.getcwd() 去讀 .ini / arm.yaml，
# 所以先切到 workspace 根目錄再 import，讓測試不必依賴你從哪裡執行 pytest。
_PREV_CWD = os.getcwd()
os.chdir(_WS_ROOT)
try:
    import hiwin_control.arm_controler_how as arm
finally:
    os.chdir(_PREV_CWD)

# 防呆：萬一還是抓到別份，直接爆掉，不要讓測試結果騙人。
assert os.path.abspath(arm.__file__).startswith(_PKG_SRC), (
    '測試匯入到錯的程式碼：' + arm.__file__ + '，預期在 ' + _PKG_SRC)


@pytest.fixture
def node():
    rclpy.init()
    controller = arm.Hiwin_Controller()
    yield controller
    controller.destroy_node()
    rclpy.shutdown()


def make_response(arm_state, digital_state=1):
    """手臂會回的 response。arm_state: 1=IDLE, 2=RUNNING, 3=HOLD, 4=DELAY."""
    res = RobotCommand.Response()
    res.arm_state = arm_state
    res.digital_state = digital_state
    res.current_position = [0.0, 332.87, 425.164, -179.999, -0.002, 89.644]
    return res


def test_reaches_strategy_when_arm_reports_running(node):
    """從 LOCK_INFO 出發，不管手臂回報什麼 arm_state，
    狀態機都必須走到 STRATEGY，中途不得回傳 None（無聲結束）。
    """
    if arm.USE_DYNAMIC_CALI:
        pytest.skip('USE_DYNAMIC_CALI=True 會進入需要即時影像的視覺伺服迴圈，'
                    '這條路徑不能離線跑')

    # 假的手臂：一律回報 RUNNING(2)。這是手臂合法會回的值 ——
    # server 對 DIGITAL_OUTPUT 不做輪詢，回的是觸發後那一瞬間的狀態。
    sent = []

    def fake_call_hiwin(req):
        sent.append(req)
        return make_response(arm_state=2)

    node.call_hiwin = fake_call_hiwin

    # 假的 YOLO 結果，用 2026-07-27 那次實跑 log 裡的座標：
    # target and cue: [[823.0, 877.5], [526.0, 558.5]]
    node.all_ball_pose = [823.0, 877.5, 526.0, 558.5]
    node.all_label = ['9', 'white']

    trace = [arm.States.LOCK_INFO]
    state = arm.States.LOCK_INFO
    for _ in range(10):
        state = node._state_machine(state)
        trace.append(state)
        assert state is not None, (
            '狀態機無聲結束（_main_loop 會直接 break）。走過的路徑：'
            + ' -> '.join(s.name if s else 'None' for s in trace))
        if state == arm.States.STRATEGY:
            break

    assert state == arm.States.STRATEGY, (
        '沒有走到 STRATEGY。走過的路徑：'
        + ' -> '.join(s.name for s in trace))
