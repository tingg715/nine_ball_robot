#!/usr/bin/env python3
"""
Interactive Pool Aiming UI (Tkinter)

Shows a top-down pool table after ball positions have been calibrated.
Lets the user:
  1. Select which pot (pocket) to aim for (0-5)
  2. Adjust the hit-ball vector angle with a slider
  3. Confirm to return the chosen vector back to the state machine
"""

import tkinter as tk
from tkinter import ttk
import math
import numpy as np

# ── Import table geometry from the strategy module ──────────────────────────
import hiwin_control.nine_ball_strat as strat

# Table / ball / hole constants (mm, in arm coordinate space)
TABLE_W = strat.tablewidth   # 627
TABLE_H = strat.tableheight  # 304
BALL_R  = strat.r            # 16
HOLE_R  = strat.rb           # 20

# Pot positions & aim-points (lists of length 6)
HOLE_X = strat.holex
HOLE_Y = strat.holey
AIM_X  = strat.aimpointx
AIM_Y  = strat.aimpointy

# ── Canvas layout constants ─────────────────────────────────────────────────
CANVAS_PAD = 60          # px padding around the table drawing
CANVAS_SCALE = 1.2       # px-per-mm  (627mm × 1.2 ≈ 752 px)

POT_NAMES = [
    "Pot 0 (top-left)",
    "Pot 1 (bot-left)",
    "Pot 2 (bot-mid)",
    "Pot 3 (bot-right)",
    "Pot 4 (top-right)",
    "Pot 5 (top-mid)",
]

# Colours
CLR_TABLE   = "#0a6e2e"
CLR_RAIL    = "#4a2f15"
CLR_HOLE    = "#111111"
CLR_CUE     = "#f0f0f0"
CLR_TARGET  = "#e8c33a"
CLR_OBJ     = "#2277cc"
CLR_ARROW   = "#ff3355"
CLR_AIM_LN  = "#ffaa00"
CLR_BG      = "#1a1a2e"
CLR_PANEL   = "#16213e"
CLR_TEXT    = "#e0e0e0"
CLR_ACCENT  = "#0f3460"
CLR_BTN     = "#e94560"


def _arm_to_canvas(ax, ay, origin_x, origin_y, table_bounds):
    """Convert arm-coordinate (mm) to canvas pixel position.

    The arm coordinate system has an arbitrary origin.  We map the table
    bounding box (derived from the pot positions) onto the canvas rectangle.

    Parameters
    ----------
    ax, ay : float          – point in arm coords (mm)
    origin_x, origin_y : float – arm-coord of table bottom-left corner
    table_bounds : (w, h)   – arm-coord table size (mm)
    """
    # normalise to 0..1 within the table
    nx = (ax - origin_x) / table_bounds[0]
    ny = (ay - origin_y) / table_bounds[1]
    # canvas pixel (y is flipped: arm-y increases upward, canvas-y downward)
    cx = CANVAS_PAD + nx * TABLE_W * CANVAS_SCALE
    cy = CANVAS_PAD + (1.0 - ny) * TABLE_H * CANVAS_SCALE
    return cx, cy


def show_aim_ui(
    object_ball_x: list,
    object_ball_y: list,
    cue_x: float,
    cue_y: float,
) -> dict:
    """Launch the aiming UI (blocking).

    Parameters
    ----------
    object_ball_x, object_ball_y : list[float]
        Calibrated object-ball positions in arm coords (mm).
    cue_x, cue_y : float
        Calibrated cue-ball position in arm coords (mm).

    Returns
    -------
    dict  with keys:
        pot          – selected pot index (0-5)
        vx, vy       – adjusted hit vector components (arm-mm)
        hitpoint_x/y – cue-stick contact point on cue ball (arm-mm)
        score        – 0  (manual override)
        obstacle     – 0  (user is responsible)
    """

    # ── Determine table bounding box from pot positions ─────────────
    all_x = list(HOLE_X)
    all_y = list(HOLE_Y)
    tbl_min_x = min(all_x)
    tbl_max_x = max(all_x)
    tbl_min_y = min(all_y)
    tbl_max_y = max(all_y)
    tbl_w = tbl_max_x - tbl_min_x
    tbl_h = tbl_max_y - tbl_min_y

    def a2c(ax, ay):
        return _arm_to_canvas(ax, ay, tbl_min_x, tbl_min_y, (tbl_w, tbl_h))

    # ── Result container ────────────────────────────────────────────
    result = {}

    # ── Build window ────────────────────────────────────────────────
    root = tk.Tk()
    root.title("Pool Aim Selector")
    root.configure(bg=CLR_BG)
    root.resizable(False, False)

    canvas_w = int(TABLE_W * CANVAS_SCALE + 2 * CANVAS_PAD)
    canvas_h = int(TABLE_H * CANVAS_SCALE + 2 * CANVAS_PAD)

    # ── Style ───────────────────────────────────────────────────────
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("TFrame", background=CLR_PANEL)
    style.configure("TLabel", background=CLR_PANEL, foreground=CLR_TEXT,
                     font=("Helvetica", 11))
    style.configure("Title.TLabel", background=CLR_BG, foreground=CLR_TEXT,
                     font=("Helvetica", 16, "bold"))
    style.configure("TRadiobutton", background=CLR_PANEL, foreground=CLR_TEXT,
                     font=("Helvetica", 10))
    style.configure("Accent.TButton", background=CLR_BTN, foreground="white",
                     font=("Helvetica", 13, "bold"), padding=(20, 10))
    style.map("Accent.TButton",
              background=[("active", "#c73650"), ("pressed", "#a82040")])

    # Title
    ttk.Label(root, text="🎱  Select Pot & Adjust Aim", style="Title.TLabel"
              ).pack(pady=(12, 4))

    # Main frame: canvas + controls side by side
    main_frame = tk.Frame(root, bg=CLR_BG)
    main_frame.pack(padx=10, pady=6)

    # ── Canvas ──────────────────────────────────────────────────────
    canvas = tk.Canvas(main_frame, width=canvas_w, height=canvas_h,
                       bg=CLR_BG, highlightthickness=0)
    canvas.grid(row=0, column=0, padx=(0, 10))

    # ── Right-side controls panel ───────────────────────────────────
    ctrl = ttk.Frame(main_frame, style="TFrame", padding=12)
    ctrl.grid(row=0, column=1, sticky="ns")

    # Pot selection
    ttk.Label(ctrl, text="Target Pot:").pack(anchor="w", pady=(0, 4))
    pot_var = tk.IntVar(value=0)

    for idx, name in enumerate(POT_NAMES):
        ttk.Radiobutton(ctrl, text=name, variable=pot_var, value=idx,
                         command=lambda: _redraw()
                         ).pack(anchor="w", padx=8, pady=1)

    ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)

    # Angle slider
    ttk.Label(ctrl, text="Angle Offset (°):").pack(anchor="w", pady=(0, 2))
    angle_var = tk.DoubleVar(value=0.0)
    angle_slider = tk.Scale(ctrl, from_=-30, to=30, resolution=0.01,
                             orient="horizontal", variable=angle_var,
                             length=200, bg=CLR_PANEL, fg=CLR_TEXT,
                             troughcolor=CLR_ACCENT, highlightthickness=0,
                             command=lambda _: _redraw())
    angle_slider.pack(anchor="w", padx=8, pady=(0, 4))

    angle_label = ttk.Label(ctrl, text="0.0°")
    angle_label.pack(anchor="w", padx=8)

    # Direct angle text entry
    angle_entry_frame = tk.Frame(ctrl, bg=CLR_PANEL)
    angle_entry_frame.pack(anchor="w", padx=8, pady=(6, 0))
    ttk.Label(angle_entry_frame, text="Type angle:").pack(side="left")
    angle_entry_var = tk.StringVar(value="0.0")
    angle_entry = ttk.Entry(angle_entry_frame, textvariable=angle_entry_var, width=8)
    angle_entry.pack(side="left", padx=(6, 0))

    def _apply_angle_entry(*_):
        try:
            val = float(angle_entry_var.get())
            val = max(-30.0, min(30.0, val))
            angle_var.set(val)
            _redraw()
        except ValueError:
            pass

    angle_entry.bind("<Return>", _apply_angle_entry)
    angle_entry.bind("<FocusOut>", _apply_angle_entry)

    def _sync_entry_from_slider(_=None):
        angle_entry_var.set(f"{angle_var.get():.2f}")
        _redraw()

    angle_slider.configure(command=_sync_entry_from_slider)

    ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)

    # Info labels
    info_pot_label  = ttk.Label(ctrl, text="Pot: 0")
    info_pot_label.pack(anchor="w")
    info_vec_label  = ttk.Label(ctrl, text="Vector: (0, 0)")
    info_vec_label.pack(anchor="w")
    info_hit_label  = ttk.Label(ctrl, text="Hit pt: (0, 0)")
    info_hit_label.pack(anchor="w")
    info_obs_label  = ttk.Label(ctrl, text="Obstacle: —")
    info_obs_label.pack(anchor="w")

    ttk.Separator(ctrl, orient="horizontal").pack(fill="x", pady=10)

    # ── Obstacle detection (same logic as nine_ball_strat.main) ────
    def _check_obstacle(vx, vy, hx, hy):
        """Check end-effector collision & hitpoint out-of-bounds.
        Returns 1 if obstacle present, 0 if clear."""
        out_flag = strat.outofbound(hx, hy)
        poly1, poly2, poly3, poly4 = strat.end_effector_mask(
            cue_x, cue_y, vx, vy)
        obs_flag, _ = strat.check_obstacle(
            poly1, poly2, poly3, poly4,
            object_ball_x, object_ball_y)
        return 1 if (out_flag or obs_flag) else 0

    # Confirm button
    def _on_confirm():
        pot = pot_var.get()
        offset_deg = angle_var.get()
        vx, vy, hx, hy = _compute_vector(pot, offset_deg)
        obstacle = _check_obstacle(vx, vy, hx, hy)
        result.update({
            "pot": pot,
            "vx": vx,
            "vy": vy,
            "hitpoint_x": hx,
            "hitpoint_y": hy,
            "score": 0,
            "obstacle": obstacle,
        })
        root.destroy()

    confirm_btn = ttk.Button(ctrl, text="✔  Confirm & Hit",
                              style="Accent.TButton", command=_on_confirm)
    confirm_btn.pack(pady=(8, 0), fill="x")

    # ── Table boundary (arm coords) for wall-bounce clipping ───────
    wall_left   = min(HOLE_X)
    wall_right  = max(HOLE_X)
    wall_bottom = min(HOLE_Y)
    wall_top    = max(HOLE_Y)

    # ── Helper: compute hit vector ──────────────────────────────────
    def _compute_vector(pot_idx, offset_deg):
        """Return (vx, vy, hitpoint_x, hitpoint_y) in arm coords."""
        aim_x = AIM_X[pot_idx]
        aim_y = AIM_Y[pot_idx]

        # direction from target ball toward aim-point
        target_vx = aim_x - object_ball_x[0]
        target_vy = aim_y - object_ball_y[0]

        # hitpoint on the target ball surface (opposite to aim direction)
        hp_x, hp_y = strat.findhitpoint(
            object_ball_x[0], object_ball_y[0], target_vx, target_vy)

        # base vector from cue to hitpoint
        base_vx = hp_x - cue_x
        base_vy = hp_y - cue_y

        # rotate by offset
        rad = math.radians(offset_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        adj_vx = base_vx * cos_a - base_vy * sin_a
        adj_vy = base_vx * sin_a + base_vy * cos_a

        # recompute hitpoint from adjusted vector
        final_hp_x, final_hp_y = strat.hitpoint(
            cue_x, cue_y, adj_vx, adj_vy)

        return adj_vx, adj_vy, final_hp_x, final_hp_y

    def _compute_trajectory(pot_idx, offset_deg):
        """Compute full ball trajectory segments for live preview.

        Returns
        -------
        cue_path    : list of (x,y) arm-coord waypoints  – cue ball before contact
        target_path : list of (x,y) arm-coord waypoints  – target ball after contact
        cue_defl    : list of (x,y) arm-coord waypoints  – cue ball deflection after contact
        contact_pt  : (x,y) – where cue ball contacts target ball surface
        """
        aim_x = AIM_X[pot_idx]
        aim_y = AIM_Y[pot_idx]

        # Target ball → aim-point vector (direction target will travel)
        target_vx = aim_x - object_ball_x[0]
        target_vy = aim_y - object_ball_y[0]
        target_vlen = math.sqrt(target_vx**2 + target_vy**2)

        # Contact point on target ball surface
        contact_x, contact_y = strat.findhitpoint(
            object_ball_x[0], object_ball_y[0], target_vx, target_vy)

        # Cue → contact vector (base)
        base_vx = contact_x - cue_x
        base_vy = contact_y - cue_y

        # Apply angle offset
        rad = math.radians(offset_deg)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        adj_vx = base_vx * cos_a - base_vy * sin_a
        adj_vy = base_vx * sin_a + base_vy * cos_a
        adj_vlen = math.sqrt(adj_vx**2 + adj_vy**2)

        if adj_vlen < 1e-6:
            return [(cue_x, cue_y)], [(object_ball_x[0], object_ball_y[0])], [], (cue_x, cue_y)

        # Unit vector of cue's travel direction
        cue_ux = adj_vx / adj_vlen
        cue_uy = adj_vy / adj_vlen

        # ── Cue ball path: from cue to where it would contact ──────
        # Find where cue ball centre will be at moment of contact (2r from target centre)
        # along the adjusted direction, find intersection with circle of radius 2r around target
        # For simplicity, project cue direction onto target and find contact
        dx = object_ball_x[0] - cue_x
        dy = object_ball_y[0] - cue_y
        # project (dx, dy) onto cue direction
        proj = dx * cue_ux + dy * cue_uy
        # perpendicular distance
        perp_sq = dx**2 + dy**2 - proj**2
        contact_r = 2 * BALL_R
        if perp_sq < contact_r**2 and proj > 0:
            # distance along cue direction to contact
            hit_dist = proj - math.sqrt(contact_r**2 - perp_sq)
            cue_contact_x = cue_x + cue_ux * hit_dist
            cue_contact_y = cue_y + cue_uy * hit_dist
        else:
            # no real contact — just extend cue direction
            hit_dist = adj_vlen
            cue_contact_x = cue_x + adj_vx
            cue_contact_y = cue_y + adj_vy

        cue_path = [(cue_x, cue_y), (cue_contact_x, cue_contact_y)]

        # ── Target ball path after contact ─────────────────────────
        # Direction: from cue contact point toward target ball centre
        impact_vx = object_ball_x[0] - cue_contact_x
        impact_vy = object_ball_y[0] - cue_contact_y
        impact_vlen = math.sqrt(impact_vx**2 + impact_vy**2)
        if impact_vlen < 1e-6:
            impact_ux, impact_uy = cue_ux, cue_uy
        else:
            impact_ux = impact_vx / impact_vlen
            impact_uy = impact_vy / impact_vlen

        # Extend target ball trajectory with wall bounces
        target_path = _trace_with_bounces(
            object_ball_x[0], object_ball_y[0], impact_ux, impact_uy, 600)

        # ── Cue ball deflection after contact ──────────────────────
        # In ideal pool physics: cue ball deflects at 90° to the target ball path
        # Deflection = cue velocity component perpendicular to impact direction
        dot = cue_ux * impact_ux + cue_uy * impact_uy
        defl_vx = cue_ux - dot * impact_ux
        defl_vy = cue_uy - dot * impact_uy
        defl_len = math.sqrt(defl_vx**2 + defl_vy**2)
        if defl_len > 1e-6:
            defl_ux = defl_vx / defl_len
            defl_uy = defl_vy / defl_len
            cue_defl = _trace_with_bounces(
                cue_contact_x, cue_contact_y, defl_ux, defl_uy, 300)
        else:
            # head-on collision: cue ball stops
            cue_defl = [(cue_contact_x, cue_contact_y)]

        return cue_path, target_path, cue_defl, (cue_contact_x, cue_contact_y)

    def _trace_with_bounces(start_x, start_y, ux, uy, max_dist, max_bounces=2):
        """Trace a ball path with wall bounces. Returns list of (x,y) waypoints."""
        pts = [(start_x, start_y)]
        cx, cy = start_x, start_y
        dx, dy = ux, uy
        remaining = max_dist

        for _ in range(max_bounces + 1):
            if remaining <= 0:
                break

            # Find nearest wall intersection
            best_t = remaining  # default: travel full remaining distance
            hit_wall = None  # 'left', 'right', 'top', 'bottom'

            # Left wall
            if dx < 0 and cx > wall_left:
                t = (wall_left - cx) / dx
                if 0 < t < best_t:
                    best_t = t
                    hit_wall = 'left'
            # Right wall
            if dx > 0 and cx < wall_right:
                t = (wall_right - cx) / dx
                if 0 < t < best_t:
                    best_t = t
                    hit_wall = 'right'
            # Bottom wall
            if dy < 0 and cy > wall_bottom:
                t = (wall_bottom - cy) / dy
                if 0 < t < best_t:
                    best_t = t
                    hit_wall = 'bottom'
            # Top wall
            if dy > 0 and cy < wall_top:
                t = (wall_top - cy) / dy
                if 0 < t < best_t:
                    best_t = t
                    hit_wall = 'top'

            # Move to hit point
            cx += dx * best_t
            cy += dy * best_t
            remaining -= best_t
            pts.append((cx, cy))

            if hit_wall is None or remaining <= 0:
                break

            # Reflect direction
            if hit_wall in ('left', 'right'):
                dx = -dx
            else:
                dy = -dy

        return pts

    # ── Trajectory colour constants ─────────────────────────────────
    CLR_CUE_PATH    = "#ff3355"     # red – cue ball approach
    CLR_TARGET_PATH = "#ffaa00"     # orange – target ball after contact
    CLR_CUE_DEFL    = "#00ccff"     # cyan – cue ball deflection
    CLR_CONTACT     = "#ffffff"     # white – contact point marker
    CLR_GHOST       = "#ffffff30"   # faint ghost ball at contact

    # ── Draw everything ─────────────────────────────────────────────
    def _redraw():
        canvas.delete("all")

        # -- Table felt
        x0, y0 = CANVAS_PAD, CANVAS_PAD
        x1 = CANVAS_PAD + TABLE_W * CANVAS_SCALE
        y1 = CANVAS_PAD + TABLE_H * CANVAS_SCALE
        # Rail
        rail_pad = 12
        canvas.create_rectangle(x0 - rail_pad, y0 - rail_pad,
                                 x1 + rail_pad, y1 + rail_pad,
                                 fill=CLR_RAIL, outline="#2a1a08", width=3)
        # Felt
        canvas.create_rectangle(x0, y0, x1, y1,
                                 fill=CLR_TABLE, outline=CLR_TABLE)

        # -- Holes
        for i in range(6):
            cx, cy = a2c(HOLE_X[i], HOLE_Y[i])
            hr = HOLE_R * CANVAS_SCALE
            canvas.create_oval(cx - hr, cy - hr, cx + hr, cy + hr,
                                fill=CLR_HOLE, outline="#333333", width=1)
            canvas.create_text(cx, cy + hr + 10, text=str(i),
                                fill="#aaaaaa", font=("Helvetica", 9))

        # -- Aim points (small dots)
        for i in range(6):
            cx, cy = a2c(AIM_X[i], AIM_Y[i])
            ar = 3
            canvas.create_oval(cx - ar, cy - ar, cx + ar, cy + ar,
                                fill="#ff6666", outline="")

        # -- Object balls
        for i in range(len(object_ball_x)):
            cx, cy = a2c(object_ball_x[i], object_ball_y[i])
            br = BALL_R * CANVAS_SCALE
            color = CLR_TARGET if i == 0 else CLR_OBJ
            canvas.create_oval(cx - br, cy - br, cx + br, cy + br,
                                fill=color, outline="white", width=1)
            canvas.create_text(cx, cy, text=str(i),
                                fill="white", font=("Helvetica", 9, "bold"))

        # -- Cue ball
        ccx, ccy = a2c(cue_x, cue_y)
        br = BALL_R * CANVAS_SCALE
        canvas.create_oval(ccx - br, ccy - br, ccx + br, ccy + br,
                            fill=CLR_CUE, outline="#999999", width=2)
        canvas.create_text(ccx, ccy, text="C",
                            fill="#333", font=("Helvetica", 9, "bold"))

        # -- Selected pot highlight
        pot = pot_var.get()
        px, py = a2c(HOLE_X[pot], HOLE_Y[pot])
        hr2 = HOLE_R * CANVAS_SCALE + 5
        canvas.create_oval(px - hr2, py - hr2, px + hr2, py + hr2,
                            fill="", outline=CLR_BTN, width=3)

        # ── Live trajectory ─────────────────────────────────────────
        offset_deg = angle_var.get()
        vx, vy, hx, hy = _compute_vector(pot, offset_deg)
        cue_path, target_path, cue_defl, contact_pt = _compute_trajectory(pot, offset_deg)

        # Helper: draw polyline from arm-coord waypoints
        def _draw_path(pts, color, width=2, dash=None, arrow=True):
            if len(pts) < 2:
                return
            canvas_pts = []
            for (ax, ay) in pts:
                cx, cy = a2c(ax, ay)
                canvas_pts.extend([cx, cy])
            kw = dict(fill=color, width=width, smooth=False)
            if dash:
                kw['dash'] = dash
            if arrow:
                kw['arrow'] = 'last'
                kw['arrowshape'] = (10, 13, 4)
            canvas.create_line(*canvas_pts, **kw)

        # 1) Cue ball approach (solid red arrow)
        _draw_path(cue_path, CLR_CUE_PATH, width=3)

        # 2) Target ball trajectory after impact (solid orange arrow, with bounces)
        _draw_path(target_path, CLR_TARGET_PATH, width=2)

        # 3) Cue ball deflection after impact (dashed cyan)
        _draw_path(cue_defl, CLR_CUE_DEFL, width=2, dash=(8, 5))

        # 4) Contact point marker
        cp_cx, cp_cy = a2c(contact_pt[0], contact_pt[1])
        canvas.create_oval(cp_cx - 5, cp_cy - 5, cp_cx + 5, cp_cy + 5,
                            fill="", outline=CLR_CONTACT, width=2)

        # 5) Ghost cue ball at contact position
        gbr = BALL_R * CANVAS_SCALE
        canvas.create_oval(cp_cx - gbr, cp_cy - gbr, cp_cx + gbr, cp_cy + gbr,
                            fill="", outline="#ffffff", width=1, dash=(3, 3))

        # -- Obstacle check (live)
        obstacle = _check_obstacle(vx, vy, hx, hy)

        # -- Update info labels
        angle_label.configure(text=f"{offset_deg:+.2f}°")
        info_pot_label.configure(text=f"Pot: {pot}  ({POT_NAMES[pot]})")
        info_vec_label.configure(text=f"Vector: ({vx:.1f}, {vy:.1f})")
        info_hit_label.configure(text=f"Hit pt: ({hx:.1f}, {hy:.1f})")
        if obstacle:
            info_obs_label.configure(text="Obstacle: ⚠ YES", foreground="#ff5555")
        else:
            info_obs_label.configure(text="Obstacle: ✓ Clear", foreground="#55ff55")

    # Initial draw
    _redraw()

    # ── Run ──────────────────────────────────────────────────────────
    root.update_idletasks()
    # Centre window on screen
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    ww = root.winfo_width()
    wh = root.winfo_height()
    root.geometry(f"+{(sw-ww)//2}+{(sh-wh)//2}")

    root.mainloop()

    # If user closed window without confirming, return a default
    if not result:
        # fallback: straight shot at pot 0
        vx, vy, hx, hy = _compute_vector(0, 0.0)
        obstacle = _check_obstacle(vx, vy, hx, hy)
        result.update({
            "pot": 0,
            "vx": vx,
            "vy": vy,
            "hitpoint_x": hx,
            "hitpoint_y": hy,
            "score": 0,
            "obstacle": obstacle,
        })

    return result
