#!/usr/bin/env python3
"""
Billiards Shot Planner UI — Redesigned
Logic flow:
  1. Watch all balls from YOLO → freeze positions
  2. Find white ball + target ball (user selects)
  3. User selects target pocket
  4. Calculate best angle, force, spin to pot target ball
  5. Show white ball path + target ball path
  6. Send shot to robot → arm hits ball
  7. Robot returns to photo pose → repeat

ROS2 Topics:
  Subscribe: /center_data_coords  (Float64MultiArray) — ball pixel positions
             /center_data_labels  (String)            — ball labels
  Publish:   /strategy_hitpoint   (Float64MultiArray) — [hit_x, hit_y, angle, force]
"""

import math
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, String

# ── Canvas dimensions ─────────────────────────────────
W, H   = 680, 400
PAD    = 36
TW     = W - PAD * 2
TH     = H - PAD * 2
BR     = 11
PR     = 12

# ── Real table size mm ────────────────────────────────
TABLE_REAL_W = 2540.0
TABLE_REAL_H = 1270.0
YOLO_W = 1920
YOLO_H = 1080

POCKET_POSITIONS = [
    (PAD,   PAD,   'TL', 'Top-Left'),
    (W//2,  PAD,   'TM', 'Top-Mid'),
    (W-PAD, PAD,   'TR', 'Top-Right'),
    (PAD,   H-PAD, 'BL', 'Bot-Left'),
    (W//2,  H-PAD, 'BM', 'Bot-Mid'),
    (W-PAD, H-PAD, 'BR', 'Bot-Right'),
]

BALL_COLORS = {
    'white': '#ffffff', 'cue': '#ffffff', 'w': '#ffffff',
    '1': '#f5e642', '2': '#1a3ab5', '3': '#e74c3c',
    '4': '#6e2da8', '5': '#e67e22', '6': '#1a7a3a',
    '7': '#8b1a1a', '8': '#111111', '9': '#f5e642',
    'black': '#111111',
}

STATE_WATCH   = 'WATCHING'
STATE_FROZEN  = 'FROZEN'
STATE_SELECT  = 'SELECT_TARGET'
STATE_PREVIEW = 'PREVIEW'
STATE_HIT     = 'HITTING'


# ══════════════════════════════════════════════════════
#  Physics
# ══════════════════════════════════════════════════════
def simulate_path(sx, sy, vx, vy, steps, bounce=True):
    path = [(sx, sy)]
    x, y, dvx, dvy = sx, sy, vx, vy
    for _ in range(steps):
        x += dvx; y += dvy
        if bounce:
            if x <= PAD+BR:   dvx =  abs(dvx); x = PAD+BR
            if x >= W-PAD-BR: dvx = -abs(dvx); x = W-PAD-BR
            if y <= PAD+BR:   dvy =  abs(dvy); y = PAD+BR
            if y >= H-PAD-BR: dvy = -abs(dvy); y = H-PAD-BR
        else:
            if not (PAD <= x <= W-PAD and PAD <= y <= H-PAD):
                break
        path.append((x, y))
    return path


def compute_collision(wvx, wvy, hx, hy, tx, ty):
    dx, dy = tx-hx, ty-hy
    dist = math.hypot(dx, dy)
    if dist == 0:
        return (wvx, wvy), (0.0, 0.0)
    nx, ny = dx/dist, dy/dist
    dot = wvx*nx + wvy*ny
    tvx, tvy = dot*nx, dot*ny
    rvx, rvy = wvx-tvx, wvy-tvy
    return (tvx, tvy), (rvx, rvy)


def calc_hit_angle(wx, wy, tx, ty, px, py):
    target_to_pocket = math.atan2(py-ty, px-tx)
    ghost_x = tx - math.cos(target_to_pocket)*BR*2
    ghost_y = ty - math.sin(target_to_pocket)*BR*2
    hit_angle = math.degrees(math.atan2(ghost_y-wy, ghost_x-wx)) % 360
    return hit_angle, ghost_x, ghost_y


def pixel_to_mm(px, py):
    return round((px-PAD)/TW*TABLE_REAL_W, 1), round((py-PAD)/TH*TABLE_REAL_H, 1)


def yolo_to_canvas(yolo_x, yolo_y):
    return PAD + (yolo_x/YOLO_W)*TW, PAD + (yolo_y/YOLO_H)*TH


# ══════════════════════════════════════════════════════
#  ROS2 Node
# ══════════════════════════════════════════════════════
class PlannerNode(Node):
    def __init__(self):
        super().__init__('billiards_planner')
        self.ball_coords = []
        self.ball_labels = []
        self._lock = threading.Lock()
        self.create_subscription(Float64MultiArray, 'center_data_coords', self._cb_coords, 10)
        self.create_subscription(String, 'center_data_labels', self._cb_labels, 10)
        self.pub_hit = self.create_publisher(Float64MultiArray, 'strategy_hitpoint', 10)
        self.get_logger().info('Billiards planner started')

    def _cb_coords(self, msg):
        with self._lock:
            self.ball_coords = list(msg.data)

    def _cb_labels(self, msg):
        with self._lock:
            self.ball_labels = eval(msg.data)

    def get_balls(self):
        with self._lock:
            coords = list(self.ball_coords)
            labels = list(self.ball_labels)
        balls = []
        for i in range(0, len(coords)-1, 2):
            idx = i//2
            label = labels[idx] if idx < len(labels) else str(idx)
            cx, cy = yolo_to_canvas(coords[i], coords[i+1])
            balls.append({'label': label, 'x': cx, 'y': cy})
        return balls

    def publish_shot(self, wx, wy, angle_deg, force):
        msg = Float64MultiArray()
        rx, ry = pixel_to_mm(wx, wy)
        msg.data = [rx, ry, float(angle_deg), float(force)]
        self.pub_hit.publish(msg)
        self.get_logger().info(f'Shot: ({rx},{ry})mm angle={angle_deg:.1f} force={force}')


# ══════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════
class BilliardsUI:
    def __init__(self, root, node):
        self.root  = root
        self.node  = node
        root.title('Billiards Shot Planner')
        root.configure(bg='#1a1a1a')
        root.resizable(False, False)

        self.state         = STATE_WATCH
        self.frozen_balls  = []
        self.white_ball    = None
        self.target_ball   = None
        self.target_pocket = None
        self.best_angle    = 0.0
        self.ghost_pos     = None
        self._drag_white   = False
        self._watching     = True

        self._build_ui()
        self._refresh()

    def _build_ui(self):
        top = tk.Frame(self.root, bg='#111', height=36)
        top.pack(fill='x')
        self.lbl_state = tk.Label(top, text='WATCHING BALLS',
            bg='#111', fg='#27ae60', font=('Helvetica', 11, 'bold'), padx=12)
        self.lbl_state.pack(side='left', pady=6)
        self.lbl_hint = tk.Label(top, text='Waiting for YOLO...',
            bg='#111', fg='#888', font=('Helvetica', 10), padx=8)
        self.lbl_hint.pack(side='left', pady=6)

        main = tk.Frame(self.root, bg='#1a1a1a')
        main.pack(fill='both', expand=True, padx=8, pady=8)

        self.canvas = tk.Canvas(main, width=W, height=H,
            bg='#155c2c', highlightthickness=0, cursor='crosshair')
        self.canvas.grid(row=0, column=0, rowspan=30, padx=(0,10))
        self.canvas.bind('<ButtonPress-1>',   self._on_click)
        self.canvas.bind('<B1-Motion>',       self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)

        rp = tk.Frame(main, bg='#1a1a1a', width=200)
        rp.grid(row=0, column=1, sticky='nw')

        def sec(t, r):
            tk.Label(rp, text=t, bg='#1a1a1a', fg='#555',
                     font=('Helvetica', 9)).grid(row=r, column=0, columnspan=2,
                                                  sticky='w', pady=(10,2))
        def sldr(r, var, lo, hi, val):
            s = tk.Scale(rp, variable=var, from_=lo, to=hi,
                         orient='horizontal', length=170,
                         bg='#222', fg='#eee', troughcolor='#444',
                         highlightthickness=0,
                         command=lambda _: self._on_param_change())
            s.set(val)
            s.grid(row=r, column=0, columnspan=2, pady=2)

        sec('FORCE', 0); self.v_force = tk.IntVar(value=6); sldr(1, self.v_force, 1, 10, 6)
        sec('SPIN',  2); self.v_spin  = tk.IntVar(value=0); sldr(3, self.v_spin, -5, 5, 0)

        self.v_bounce = tk.BooleanVar(value=True)
        tk.Checkbutton(rp, text='Show bounces', variable=self.v_bounce,
            bg='#1a1a1a', fg='#aaa', selectcolor='#333',
            activebackground='#1a1a1a', activeforeground='#eee',
            command=self._draw).grid(row=4, column=0, columnspan=2, sticky='w', pady=4)

        ttk.Separator(rp).grid(row=5, column=0, columnspan=2, sticky='ew', pady=6)

        sec('TARGET POCKET', 6)
        self.pocket_var = tk.StringVar(value='')
        pnames = [(n, r, c) for c,(px,py,n,full) in enumerate(POCKET_POSITIONS)
                  for r in [c//3]]
        for i,(px,py,name,full) in enumerate(POCKET_POSITIONS):
            r, c = i//3, i%3
            tk.Radiobutton(rp, text=name, variable=self.pocket_var, value=name,
                bg='#1a1a1a', fg='#aaa', selectcolor='#27ae60',
                activebackground='#1a1a1a',
                command=self._on_pocket_select).grid(
                    row=7+r, column=c, sticky='w', padx=4)

        ttk.Separator(rp).grid(row=9, column=0, columnspan=2, sticky='ew', pady=6)

        sec('SHOT INFO', 10)
        self.lbl_angle = self._info(rp, 'Angle',  11)
        self.lbl_force2= self._info(rp, 'Force',  12)
        self.lbl_spin2 = self._info(rp, 'Spin',   13)
        self.lbl_pred  = self._info(rp, 'Pocket', 14)
        self.lbl_wb    = self._info(rp, 'W-ball', 15)

        ttk.Separator(rp).grid(row=16, column=0, columnspan=2, sticky='ew', pady=6)

        self.btn_freeze = tk.Button(rp, text='Freeze Balls',
            bg='#f39c12', fg='#fff', font=('Helvetica', 10, 'bold'),
            relief='flat', padx=6, pady=5, command=self._freeze_balls)
        self.btn_freeze.grid(row=17, column=0, columnspan=2, sticky='ew', pady=2)

        self.btn_auto = tk.Button(rp, text='Auto Best Shot',
            bg='#2980b9', fg='#fff', font=('Helvetica', 10, 'bold'),
            relief='flat', padx=6, pady=5, command=self._auto_calculate)
        self.btn_auto.grid(row=18, column=0, columnspan=2, sticky='ew', pady=2)

        self.btn_send = tk.Button(rp, text='SEND TO ROBOT',
            bg='#27ae60', fg='#fff', font=('Helvetica', 11, 'bold'),
            relief='flat', padx=6, pady=8, command=self._send_shot, state='disabled')
        self.btn_send.grid(row=19, column=0, columnspan=2, sticky='ew', pady=2)

        self.btn_reset = tk.Button(rp, text='Reset / Watch Again',
            bg='#444', fg='#fff', font=('Helvetica', 10),
            relief='flat', padx=6, pady=5, command=self._reset)
        self.btn_reset.grid(row=20, column=0, columnspan=2, sticky='ew', pady=2)

        self.status = tk.Label(self.root, text='Waiting for YOLO...', anchor='w',
            bg='#111', fg='#888', font=('Courier', 9), padx=8)
        self.status.pack(fill='x', side='bottom')

    def _info(self, p, t, r):
        tk.Label(p, text=t+':', bg='#1a1a1a', fg='#555',
                 font=('Helvetica', 9)).grid(row=r, column=0, sticky='w', padx=4)
        lbl = tk.Label(p, text='--', bg='#1a1a1a', fg='#eee',
                       font=('Helvetica', 9, 'bold'))
        lbl.grid(row=r, column=1, sticky='w', padx=4)
        return lbl

    # ── State transitions ──────────────────────────────
    def _freeze_balls(self):
        live = self.node.get_balls()
        if not live:
            messagebox.showwarning('No balls', 'No YOLO detections yet.')
            return
        self.frozen_balls = live
        self._watching = False
        # Auto-find white ball
        self.white_ball = None
        for b in self.frozen_balls:
            if b['label'].lower() in ('white','cue','w'):
                self.white_ball = dict(b); break
        if not self.white_ball:
            self.white_ball = {'x': W//2-80, 'y': H//2, 'label': 'W'}
        self.state = STATE_SELECT
        self._set_state('SELECT TARGET BALL',
                        'Click on the ball you want to hit', '#f39c12')
        self._draw()
        self.status['text'] = f'Frozen {len(live)} balls. Click target ball.'

    def _on_pocket_select(self):
        name = self.pocket_var.get()
        for px, py, n, full in POCKET_POSITIONS:
            if n == name:
                self.target_pocket = (px, py, n); break
        if self.target_ball:
            self._calculate_shot()
        self._draw()

    def _on_param_change(self):
        if self.state == STATE_PREVIEW:
            self._calculate_shot()
        self._draw()

    def _calculate_shot(self):
        if not (self.white_ball and self.target_ball and self.target_pocket):
            return
        wx, wy = self.white_ball['x'], self.white_ball['y']
        tx, ty = self.target_ball['x'], self.target_ball['y']
        px, py, _ = self.target_pocket
        force  = self.v_force.get()
        spin   = self.v_spin.get()
        bounce = self.v_bounce.get()

        angle, gx, gy = calc_hit_angle(wx, wy, tx, ty, px, py)
        self.best_angle = angle
        self.ghost_pos  = (gx, gy)
        self.state      = STATE_PREVIEW

        self._set_state('PREVIEW SHOT', 'Review path then send to robot', '#2980b9')
        self.btn_send['state'] = 'normal'

        rad   = math.radians(angle)
        speed = force * 3.5
        wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
        dx, dy = gx-wx, gy-wy
        dist = max(1, math.hypot(dx, dy))
        nx_, ny_ = dx/dist, dy/dist
        spin_eff = spin*0.35
        (tvx, tvy), _ = compute_collision(wvx, wvy, gx, gy, tx, ty)
        tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
        tp = simulate_path(tx, ty, tvx, tvy, force*50, bounce)
        last = tp[-1]
        best_pd = min(math.hypot(p[0]-last[0], p[1]-last[1])
                      for p in POCKET_POSITIONS)
        best_pn = min(POCKET_POSITIONS,
                      key=lambda p: math.hypot(p[0]-last[0], p[1]-last[1]))[2]

        self.lbl_angle['text']  = f'{angle:.1f}\u00b0'
        self.lbl_force2['text'] = str(force)
        self.lbl_spin2['text']  = str(spin)
        self.lbl_pred['text']   = best_pn if best_pd < 60 else 'Miss'
        rx, ry = pixel_to_mm(wx, wy)
        self.lbl_wb['text']     = f'{rx:.0f},{ry:.0f}mm'

    def _auto_calculate(self):
        if not self.frozen_balls:
            messagebox.showinfo('Freeze first', 'Freeze ball positions first.')
            return
        if not self.target_pocket:
            messagebox.showinfo('Select pocket', 'Select a target pocket first.')
            return
        balls = [b for b in self.frozen_balls
                 if b['label'].lower() not in ('white','cue','w')]
        if not balls:
            return
        px, py, _ = self.target_pocket
        wx = self.white_ball['x'] if self.white_ball else W//2
        wy = self.white_ball['y'] if self.white_ball else H//2
        force = self.v_force.get()
        spin  = self.v_spin.get()

        best, best_d = None, 9999
        for b in balls:
            angle, gx, gy = calc_hit_angle(wx, wy, b['x'], b['y'], px, py)
            rad = math.radians(angle)
            speed = force*3.5
            wvx, wvy = math.cos(rad)*speed, math.sin(rad)*speed
            dx, dy = gx-wx, gy-wy
            dist = max(1, math.hypot(dx,dy))
            nx_, ny_ = dx/dist, dy/dist
            spin_eff = spin*0.35
            (tvx,tvy),_ = compute_collision(wvx,wvy,gx,gy,b['x'],b['y'])
            tvx += spin_eff*(-ny_); tvy += spin_eff*nx_
            tp = simulate_path(b['x'], b['y'], tvx, tvy, force*50, True)
            last = tp[-1]
            d = math.hypot(last[0]-px, last[1]-py)
            if d < best_d:
                best_d = d; best = b

        self.target_ball = dict(best)
        self._calculate_shot()
        self._draw()
        self.status['text'] = f'Auto: Ball {best["label"]} → {self.best_angle:.1f}\u00b0'

    def _send_shot(self):
        if self.state != STATE_PREVIEW:
            return
        wx = self.white_ball['x']
        wy = self.white_ball['y']
        self.node.publish_shot(wx, wy, self.best_angle, self.v_force.get())
        self.state = STATE_HIT
        self._set_state('HITTING...', 'Robot executing shot', '#e74c3c')
        self.btn_send['state'] = 'disabled'
        self.status['text'] = (f'Shot sent: angle={self.best_angle:.1f}\u00b0 '
                               f'force={self.v_force.get()}')
        self.root.after(4000, self._reset)

    def _reset(self):
        self.state         = STATE_WATCH
        self.frozen_balls  = []
        self.white_ball    = None
        self.target_ball   = None
        self.target_pocket = None
        self.ghost_pos     = None
        self._watching     = True
        self.pocket_var.set('')
        self.btn_send['state'] = 'disabled'
        for lbl in (self.lbl_angle, self.lbl_force2,
                    self.lbl_spin2, self.lbl_pred, self.lbl_wb):
            lbl['text'] = '--'
        self._set_state('WATCHING BALLS', 'Waiting for YOLO...', '#27ae60')
        self._draw()

    def _set_state(self, s, h, color):
        self.lbl_state['text'] = s
        self.lbl_state['fg']   = color
        self.lbl_hint['text']  = h

    # ── Mouse ──────────────────────────────────────────
    def _on_click(self, event):
        mx, my = event.x, event.y
        if self.state == STATE_WATCH:
            return
        if self.state == STATE_SELECT:
            for b in self.frozen_balls:
                if (b['label'].lower() not in ('white','cue','w') and
                        math.hypot(mx-b['x'], my-b['y']) <= BR+6):
                    self.target_ball = dict(b)
                    self.status['text'] = f'Target: Ball {b["label"]}'
                    if self.target_pocket:
                        self._calculate_shot()
                    else:
                        self._set_state('SELECT POCKET',
                                        'Now choose a pocket on the right',
                                        '#9b59b6')
                    self._draw()
                    return
        if (self.state in (STATE_PREVIEW, STATE_SELECT) and
                self.white_ball and
                math.hypot(mx-self.white_ball['x'],
                           my-self.white_ball['y']) <= BR+4):
            self._drag_white = True

    def _on_drag(self, event):
        if self._drag_white and self.white_ball:
            self.white_ball['x'] = max(PAD+BR+2, min(W-PAD-BR-2, event.x))
            self.white_ball['y'] = max(PAD+BR+2, min(H-PAD-BR-2, event.y))
            if self.target_ball and self.target_pocket:
                self._calculate_shot()
            self._draw()

    def _on_release(self, _):
        self._drag_white = False

    # ── Draw ───────────────────────────────────────────
    def _draw(self):
        c = self.canvas
        c.delete('all')

        # Table surface
        c.create_rectangle(PAD//2, PAD//2, W-PAD//2, H-PAD//2,
                            fill='#8B4513', outline='')
        c.create_rectangle(PAD, PAD, W-PAD, H-PAD,
                            fill='#155c2c', outline='#0d3d1c', width=1.5)
        c.create_line(W//2, PAD, W//2, H-PAD, fill='#1a7a3a', dash=(4,10))
        c.create_oval(W//2-35, H//2-35, W//2+35, H//2+35,
                      fill='', outline='#1a7a3a')

        # Pockets
        for px, py, name, full in POCKET_POSITIONS:
            sel = self.target_pocket and self.target_pocket[2] == name
            fill = '#ffff00' if sel else '#000'
            out  = '#ffcc00' if sel else '#333'
            c.create_oval(px-PR, py-PR, px+PR, py+PR,
                          fill=fill, outline=out, width=2 if sel else 1)
            c.create_text(px, py, text=name,
                          fill='#ffcc00' if sel else '#555',
                          font=('Helvetica', 6, 'bold'))

        # Shot preview paths
        if self.state == STATE_PREVIEW and self.white_ball and self.target_ball:
            self._draw_preview(c)

        # Balls
        balls = (self.frozen_balls if not self._watching
                 else self.node.get_balls())

        for b in balls:
            bx, by = b['x'], b['y']
            label  = b['label']
            color  = BALL_COLORS.get(label.lower(), '#888888')
            is_white  = label.lower() in ('white','cue','w')
            is_target = (self.target_ball and
                         abs(bx-self.target_ball['x']) < 3 and
                         abs(by-self.target_ball['y']) < 3)
            if is_target:
                c.create_oval(bx-BR-6, by-BR-6, bx+BR+6, by+BR+6,
                              fill='', outline='#ff4444', width=2, dash=(3,3))
            if is_white:
                c.create_oval(bx-BR-4, by-BR-4, bx+BR+4, by+BR+4,
                              fill='', outline='#aaaaff', dash=(2,4))
            c.create_oval(bx-BR, by-BR, bx+BR, by+BR,
                          fill=color, outline='#cccccc' if is_white else '#333333',
                          width=1.5)
            c.create_text(bx, by, text=label[:2],
                          fill='#333' if is_white else '#fff',
                          font=('Helvetica', 7, 'bold'))

        # White ball (adjustable)
        if self.white_ball and not self._watching:
            wx, wy = self.white_ball['x'], self.white_ball['y']
            c.create_oval(wx-BR-2, wy-BR-2, wx+BR+2, wy+BR+2,
                          fill='#ffffff', outline='#aaaaaa', width=1.5)
            c.create_text(wx, wy, text='W', fill='#888',
                          font=('Helvetica', 7, 'bold'))
            c.create_text(wx, wy+BR+10, text='drag',
                          fill='#888888', font=('Helvetica', 7))

        # Status
        count = len(balls)
        c.create_text(PAD+4, H-PAD-8, anchor='sw',
                      text=f'{count} balls | {self.state}',
                      fill='#1d7a3a', font=('Helvetica', 8))

        if self._watching and count == 0:
            c.create_text(W//2, H//2,
                          text='Waiting for YOLO detections...',
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
                      fill='', outline='#888888', dash=(4,4))

        # Aim lines
        c.create_line(wx, wy, gx, gy, fill='#555555', dash=(4,8))
        c.create_line(tx, ty, px, py, fill='#883333', dash=(3,6))

        dx, dy = gx-wx, gy-wy
        dist = max(1, math.hypot(dx, dy))
        nx_, ny_ = dx/dist, dy/dist
        spin_eff = spin*0.35

        steps1 = max(1, int(dist/speed))
        wp = simulate_path(wx, wy, wvx, wvy, steps1, False)
        self._path(c, wp, '#ffffff', 2, (6,4))

        (tvx, tvy), (rvx, rvy) = compute_collision(wvx, wvy, gx, gy, tx, ty)
        tvx += spin_eff*(-ny_); tvy += spin_eff*nx_

        tp = simulate_path(tx, ty, tvx, tvy, force*50, bounce)
        self._path(c, tp, '#ff5555', 2.5)
        if len(tp) > 1:
            self._arrow(c, tp[-2], tp[-1], '#ff3333')

        wp2 = simulate_path(gx, gy, rvx, rvy, force*30, bounce)
        self._path(c, wp2, '#ffff88', 1, (3,6))

        hx = wx + nx_*dist
        hy = wy + ny_*dist
        c.create_oval(hx-5, hy-5, hx+5, hy+5, fill='#ffff00', outline='')

        ax = wx + math.cos(rad)*35
        ay = wy + math.sin(rad)*35
        c.create_line(wx, wy, ax, ay, fill='#ffffff', width=2.5)
        self._arrow(c, (wx, wy), (ax, ay), '#ffffff')

    def _path(self, c, path, color, width=1.5, dash=None):
        if len(path) < 2:
            return
        flat = [v for pt in path for v in pt]
        opts = dict(fill=color, width=width, smooth=True)
        if dash:
            opts['dash'] = dash
        c.create_line(*flat, **opts)

    def _arrow(self, c, tail, tip, color):
        dx, dy = tip[0]-tail[0], tip[1]-tail[1]
        mag = math.hypot(dx, dy)
        if mag == 0:
            return
        nx, ny = dx/mag, dy/mag
        ax, ay = tip if isinstance(tip, tuple) else (tip[0], tip[1])
        c.create_polygon(ax, ay,
                         ax-nx*10+ny*4, ay-ny*10-nx*4,
                         ax-nx*10-ny*4, ay-ny*10+nx*4,
                         fill=color, outline='')

    def _refresh(self):
        if self._watching:
            self._draw()
        self.root.after(150, self._refresh)


# ══════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════
def main():
    rclpy.init()
    node = PlannerNode()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    root = tk.Tk()
    app = BilliardsUI(root, node)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()