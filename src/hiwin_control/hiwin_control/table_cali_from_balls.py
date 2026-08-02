#!/usr/bin/env python3
"""現場重定袋口座標:球貼庫邊擺,按 Enter 抓,自動擬合後寫回 arm.yaml 的 pot0~3。

為什麼是這個做法
----------------
袋口和球必須走「同一條量測通道」。球在打球時是經 homography 從像素轉出來的,
所以袋口也用 homography 量 —— 這樣 homography 若有殘留偏差,會在「球 -> 袋口」
的瞄準向量裡抵消掉。用 pendant 去戳袋口反而更糟,那是把袋口放進另一個座標系。

原理:球貼著庫邊時球心離庫邊剛好一個球半徑,所以四條「球心線」各往外推 r 就是
四條庫邊線,交點就是袋口(nine_ball_strat 的 hole)。而 hole 加回 r 得到的
aimpoint,正好等於「球貼著兩片庫時球心在哪」的實測值。

用法
----
    ros2 run center ball_coordinate_checker      # 要先跑,本工具吃它的 /ball_coordinate
    ros2 run hiwin_control table_cali_from_balls

    球貼四片庫邊擺好(長庫多擺幾顆),按 Enter 抓。球不夠就分幾輪擺,會累加。
    看殘差確認每顆都貼緊了,再按 w 寫入。

限制
----
只解決「桌子尺寸/位置不同」。如果桌面高度變了,PIXEL_TO_BASE_H 整組失效,
本工具算出來的東西也一起錯,而且不會報錯 —— 那種情況要重新擬合 homography。
"""

import json
import os
import re
import shutil
import time
from datetime import datetime
from threading import Thread

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String

# 球半徑(球徑 38mm)。跟 nine_ball_strat.py 的 r 是同一個東西,改要一起改。
BALL_R = 19.0

# 抓到的球離最近那條庫邊線超過這個距離,就當它不是貼庫的球而剔除(mm)。
MAX_DIST_TO_CUSHION = 150.0

# 兩次抓拍之間,距離小於這個值的點視為同一顆球沒動過,不重複累加(mm)。
DUPLICATE_TOL = 5.0

# /ball_coordinate 超過這個秒數沒更新就不給抓,避免用到上一輪的殘影。
MAX_MSG_AGE_SEC = 2.0

# arm.yaml 的 pot 欄位 <-> 桌子的四個角
POT_CORNERS = [
    ('pot0', 'near_left'),
    ('pot1', 'near_right'),
    ('pot2', 'far_right'),
    ('pot3', 'far_left'),
]

# 每個角在「長庫, 短庫, 朝桌內的 (u, v) 方向」
CORNER_DEF = {
    'near_left':  ('near', 'left',  (+1, +1)),
    'near_right': ('near', 'right', (-1, +1)),
    'far_right':  ('far',  'right', (-1, -1)),
    'far_left':   ('far',  'left',  (+1, -1)),
}

CUSHION_NAMES = ['near', 'far', 'left', 'right']


def fit_line(points):
    """總最小平方擬合直線。回傳 (通過點, 單位方向向量, 各點的垂直殘差)。"""
    P = np.asarray(points, dtype=np.float64)
    centre = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - centre)
    direction = Vt[0] / np.linalg.norm(Vt[0])
    normal = np.array([-direction[1], direction[0]])
    return centre, direction, (P - centre) @ normal


def intersect(p1, d1, p2, d2):
    t = np.linalg.solve(np.column_stack([d1, -d2]), p2 - p1)
    return p1 + t[0] * d1


class TableCaliFromBalls(Node):

    def __init__(self):
        super().__init__('table_cali_from_balls')

        self.declare_parameter(
            'arm_yaml',
            os.path.join(
                os.getcwd(),
                'src/hiwin_control/hiwin_control/arm.yaml',
            ),
        )
        self.arm_yaml_path = (
            self.get_parameter('arm_yaml').get_parameter_value().string_value
        )

        self.latest_balls = None
        self.latest_stamp = 0.0
        # 累加的點:{'near': [(x, y), ...], ...}
        self.captured = {name: [] for name in CUSHION_NAMES}

        self.create_subscription(
            String, '/ball_coordinate', self._ball_cb, 10
        )

        self.prior = self._load_prior()
        self.get_logger().info(f'arm.yaml: {self.arm_yaml_path}')

    # ---------------- ROS ----------------
    def _ball_cb(self, msg):
        try:
            balls = json.loads(msg.data)
        except json.JSONDecodeError as error:
            self.get_logger().warning(f'/ball_coordinate 不是合法 JSON: {error}')
            return

        if not isinstance(balls, list):
            return

        points = []
        for ball in balls:
            try:
                points.append((
                    str(ball['label']),
                    float(ball['x']),
                    float(ball['y']),
                ))
            except (TypeError, KeyError, ValueError):
                continue

        if points:
            self.latest_balls = points
            self.latest_stamp = time.time()

    # ---------------- arm.yaml ----------------
    def _load_prior(self):
        """用現有的 pot0~3 當「這顆球貼在哪條庫邊」的判斷依據。

        只用來分類,不用來算結果。就算舊值整個歪了幾十 mm 也無所謂 ——
        四條庫邊彼此相距 500mm 以上,分類不可能分錯。
        """
        with open(self.arm_yaml_path, 'r') as handle:
            data = yaml.safe_load(handle)

        corner = {
            'near_left':  np.array(data['pot0'][:2], dtype=np.float64),
            'near_right': np.array(data['pot1'][:2], dtype=np.float64),
            'far_right':  np.array(data['pot2'][:2], dtype=np.float64),
            'far_left':   np.array(data['pot3'][:2], dtype=np.float64),
        }

        u = corner['near_right'] - corner['near_left']
        u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        if v @ (corner['far_left'] - corner['near_left']) < 0:
            v = -v

        # 四條「球心線」= 庫邊線往桌內縮一個球半徑
        lines = {
            'near':  (corner['near_left'] + BALL_R * v, u),
            'far':   (corner['far_left'] - BALL_R * v, u),
            'left':  (corner['near_left'] + BALL_R * u, v),
            'right': (corner['near_right'] - BALL_R * u, v),
        }
        return {'corner': corner, 'u': u, 'v': v, 'lines': lines}

    def _classify(self, point):
        """回傳 (庫邊名稱, 到那條線的距離)。"""
        best_name, best_dist = None, float('inf')
        for name, (origin, direction) in self.prior['lines'].items():
            normal = np.array([-direction[1], direction[0]])
            dist = abs(float((np.asarray(point) - origin) @ normal))
            if dist < best_dist:
                best_name, best_dist = name, dist
        return best_name, best_dist

    # ---------------- 抓一次 ----------------
    def capture(self):
        if self.latest_balls is None:
            print('  還沒收到 /ball_coordinate。ball_coordinate_checker 有跑嗎?')
            return

        age = time.time() - self.latest_stamp
        if age > MAX_MSG_AGE_SEC:
            print(f'  /ball_coordinate 已經 {age:.1f} 秒沒更新,不抓。'
                  '檢查相機和偵測節點。')
            return

        added, rejected, duplicated = 0, [], 0
        for label, x, y in self.latest_balls:
            point = (x, y)
            name, dist = self._classify(point)

            if dist > MAX_DIST_TO_CUSHION:
                rejected.append((label, dist))
                continue

            existing = self.captured[name]
            if any(np.hypot(x - px, y - py) < DUPLICATE_TOL
                   for px, py in existing):
                duplicated += 1
                continue

            existing.append(point)
            added += 1

        print(f'  抓到 {len(self.latest_balls)} 顆,新增 {added} 顆'
              + (f',重複 {duplicated} 顆' if duplicated else '')
              + (f',剔除 {len(rejected)} 顆(離庫邊太遠)' if rejected else ''))
        for label, dist in rejected:
            print(f'    剔除: 球 {label} 離最近的庫邊 {dist:.0f}mm')

    # ---------------- 擬合 ----------------
    def solve(self):
        """回傳 (結果 dict, 問題訊息 list)。結果為 None 表示資料不足。"""
        problems = []
        for name in CUSHION_NAMES:
            count = len(self.captured[name])
            if count < 2:
                problems.append(f'{name} 只有 {count} 顆,至少要 2 顆才能擬合直線')
            elif count < 3:
                problems.append(f'{name} 只有 2 顆,沒有殘差可以檢查貼緊程度')

        if any('至少要 2 顆' in message for message in problems):
            return None, problems

        fits = {}
        for name in CUSHION_NAMES:
            fits[name] = fit_line(self.captured[name])

        # 桌面 X 軸:兩條長庫的平均方向(長庫點多,方向最可信)
        u = np.zeros(2)
        for name in ['near', 'far']:
            direction = fits[name][1]
            u = u + (direction if direction @ self.prior['u'] > 0 else -direction)
        u /= np.linalg.norm(u)
        v = np.array([-u[1], u[0]])
        if v @ self.prior['v'] < 0:
            v = -v

        # 短庫強制垂直於長庫:撞球桌是矩形製造的,自由擬合出來的不垂直
        # 是球貼緊程度不一造成的假訊號。位置仍取三顆球的擬合中心。
        lines = {name: (fits[name][0], u) for name in ['near', 'far']}
        for name in ['left', 'right']:
            lines[name] = (fits[name][0], v)

        aim, hole = {}, {}
        for corner_name, (long_c, short_c, (su, sv)) in CORNER_DEF.items():
            aim[corner_name] = intersect(*lines[long_c], *lines[short_c])
            # 球心 -> 庫邊:沿兩個「朝桌內」方向的反方向各推出一個球半徑
            hole[corner_name] = aim[corner_name] - su * BALL_R * u - sv * BALL_R * v

        return {
            'fits': fits, 'u': u, 'v': v, 'aim': aim, 'hole': hole,
        }, problems

    # ---------------- 報告 ----------------
    def report(self, result, problems):
        print()
        print('  庫邊     顆數  方向(度)   各球離擬合線(mm)')
        for name in CUSHION_NAMES:
            points = self.captured[name]
            if len(points) < 2:
                print(f'  {name:6}   {len(points):2}    ---       (顆數不足)')
                continue
            _, direction, residual = result['fits'][name] if result \
                else fit_line(points)
            angle = np.degrees(np.arctan2(direction[1], direction[0])) % 180
            marks = ' '.join(
                f'{value:+5.2f}' + ('!' if abs(value) > 3.0 else '')
                for value in residual
            )
            print(f'  {name:6}   {len(points):2}   {angle:7.3f}   {marks}')

        if result is None:
            print()
            for message in problems:
                print(f'  [不足] {message}')
            return

        u, v = result['u'], result['v']
        fits = result['fits']

        parallel = np.degrees(np.arcsin(abs(np.cross(
            fits['near'][1], fits['far'][1]))))
        print(f'\n  長庫平行度 {parallel:.4f} 度'
              + ('   <- 偏大,檢查是不是有球沒貼緊' if parallel > 0.15 else ''))
        for name in ['left', 'right']:
            deviation = 90 - np.degrees(np.arccos(
                abs(float(fits[name][1] @ u))))
            print(f'  {name:6} 對長庫的垂直度偏差 {deviation:+.4f} 度'
                  '(已強制垂直,僅供參考)')

        aim = result['aim']
        width = abs(float((aim['near_right'] - aim['near_left']) @ u))
        height = abs(float((aim['far_left'] - aim['near_left']) @ v))
        print(f'\n  球心線矩形 {width:8.2f} x {height:7.2f}'
              f'   -> 庫邊淨距 {width + 2*BALL_R:8.2f} x {height + 2*BALL_R:7.2f}')

        print(f'\n  {"":6} {"新值":>23} {"現在 arm.yaml":>23} {"差":>18}')
        for pot, corner_name in POT_CORNERS:
            new = result['hole'][corner_name]
            old = self.prior['corner'][corner_name]
            print(f'  {pot:6} ({new[0]:9.3f}, {new[1]:8.3f}) '
                  f'({old[0]:9.3f}, {old[1]:8.3f}) '
                  f'({new[0]-old[0]:+7.2f}, {new[1]-old[1]:+7.2f})')

        for message in problems:
            print(f'\n  [注意] {message}')

    # ---------------- 寫回 arm.yaml ----------------
    def write(self, result):
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        backup = f'{self.arm_yaml_path}.bak-{stamp}'
        shutil.copy2(self.arm_yaml_path, backup)

        with open(self.arm_yaml_path, 'r') as handle:
            text = handle.read()

        for pot, corner_name in POT_CORNERS:
            x, y = result['hole'][corner_name]
            # 只換 pot 底下那三行數值,key 那行(含註解)和檔案其他部分都不動。
            # z 沿用原值:所有策略檔都只讀 [0:2],z 不參與計算。
            pattern = re.compile(
                r'(^' + pot + r':[^\n]*\n)'
                r'- [^\n]*\n'
                r'- [^\n]*\n'
                r'(- [^\n]*\n)',
                re.MULTILINE,
            )
            replacement = rf'\g<1>- {x:.6f}\n- {y:.6f}\n\g<2>'
            text, count = pattern.subn(replacement, text, count=1)
            if count != 1:
                raise RuntimeError(
                    f'在 arm.yaml 裡找不到 {pot} 的三行數值,沒有寫入。'
                    f'原檔已備份在 {backup}'
                )

        with open(self.arm_yaml_path, 'w') as handle:
            handle.write(text)

        # 讀回來確認 YAML 沒被寫壞,而且值真的進去了
        with open(self.arm_yaml_path, 'r') as handle:
            check = yaml.safe_load(handle)
        for pot, corner_name in POT_CORNERS:
            expected = result['hole'][corner_name]
            actual = check[pot][:2]
            if not np.allclose(actual, expected, atol=1e-5):
                raise RuntimeError(
                    f'{pot} 寫入後對不上,請從 {backup} 還原')

        print(f'\n  已寫入 {self.arm_yaml_path}')
        print(f'  備份   {backup}')
        print('  arm.yaml 是從原始碼樹讀的,重開節點就生效,不用 rebuild。')

    # ---------------- 互動 ----------------
    def run(self):
        print()
        print('=' * 68)
        print('  袋口重定 — 球貼庫邊擺,按 Enter 抓')
        print('=' * 68)
        print('  長庫(近端/遠端)各 3 顆以上,短庫(左/右)各 2 顆以上。')
        print('  球不夠就分幾輪擺,會累加。')
        print()

        while True:
            try:
                key = input('  [Enter]抓一次  [w]寫入  [c]清除重來  [q]離開 > ')
            except EOFError:
                break

            key = key.strip().lower()

            if key == 'q':
                break

            if key == 'c':
                self.captured = {name: [] for name in CUSHION_NAMES}
                print('  已清除。')
                continue

            if key == 'w':
                result, problems = self.solve()
                if result is None:
                    print('  資料不足,不能寫入:')
                    for message in problems:
                        print(f'    {message}')
                    continue
                confirm = input('  確定寫入 arm.yaml? (yes) > ').strip().lower()
                if confirm != 'yes':
                    print('  取消。')
                    continue
                try:
                    self.write(result)
                except Exception as error:      # noqa: BLE001 - 要讓操作員看到
                    print(f'  寫入失敗: {error}')
                continue

            if key != '':
                print('  不認得的指令。')
                continue

            self.capture()
            result, problems = self.solve()
            self.report(result, problems)
            print()

        print('  結束。')
        rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TableCaliFromBalls()
    thread = Thread(target=node.run, daemon=True)
    thread.start()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
