from controller import Robot
import math
import numpy as np
import csv
import random


# ===================== ADJUSTABLE PARAMETERS =====================
# Forward distance / time thresholds (more conservative → turn earlier)
D_SAFE        = 0.95     # Normal safe distance: start turning if below this
D_HARD        = 0.55     # Very close: immediately turn hard to the open side
TTC_THRESH    = 5.0      # Time-To-Collision threshold (sec), higher → turn earlier

# Emergency narrow front window check: prevents straight collision with front post/wall
NARROW_WIN_DEG = 18      # Only check ±18° centered at the front
NARROW_ENTER   = 0.65    # Enter avoidance when narrow-front distance < this
NARROW_LEAVE   = 0.75    # Leave avoidance when narrow-front distance > this

# Speed / Gains
MAX_V = 0.95
MIN_V = 0.02
MAX_W = 1.20

KP_GAP  = 1.10           # Steering gain based on gap direction
KP_TTC  = 2.20           # Stronger turn when TTC is small
KP_WALL = 0.60           # Left/right distance balancing (keeps centered in corridors)

# Small bias when facing walls (wall-following)
EDGE_KICK_ENABLE = True     # Turn ON/OFF
EDGE_KEEP_RIGHT  = True     # True = keep right wall; False = keep left
EDGE_KICK_W      = 0.25     # Angular velocity bias (rad/s)
EDGE_SIM_THRESH  = 0.08     # Threshold for considering left/right distances “similar”
EDGE_FRONT_NEAR  = 0.90     # Considered “facing a wall” if front distance < this

# LiDAR sectors (degrees)
FRONT_DEG = 80
LF_DEG = (0, +60)
RF_DEG = (-60, 0)
GAP_SAFE = 0.95
GAP_WIN_DEG = 45         # Only search for widest gap within ±70° (avoid side/back noise)

# Stabilization parameters
SMOOTH = 0.85            # Angular smoothing filter
GAP_SMOOTH = 0.80        # Gap error low-pass filter
GAP_DEAD_BAND = 0.14     # Deadband (ignore small errors)
SIGN_STICK_EPS = 0.28    # Stickiness zone to reduce oscillation
OMEGA_SLEW = 1.0         # Angular velocity rate limit (similar to rad/s²)
V_AT_MAX_W = 0.01      # Reduce speed when turning sharply

OPEN_FRONT_DIST    = 0.9   # front distance > this = considered open
OPEN_SIDE_DIST     = 0.8   # left/right avg distance > this = open
WANDER_OMEGA_STD   = 0.4   # random step size for angular velocity
WANDER_OMEGA_DECAY = 0.92  # how much the previous turning direction persists
WANDER_OMEGA_MAX   = 0.9   # clamp |random turn rate|
WANDER_V_FACTOR    = 0.35  # forward speed factor during wandering (fraction of MAX_V)

# Quick toggles
INVERT_TURN_SIGN = False
SWAP_WHEELS = False

# Geometry (approx. TurtleBot3 Burger)
WHEEL_RADIUS = 0.033
AXLE_LENGTH  = 0.160

# Debug print
DEBUG_PRINT = True

# ===================== EXPERIMENT & COVERAGE SETTINGS =====================

# Fixed experiment duration & logging
EXPERIMENT_DURATION = 300.0   # total experiment time (seconds)
LOG_INTERVAL        = 0.1     # record every 0.1 second
WORLD_ID            = "world1"
CONTROLLER_ID       = "baseline"
RUN_ID              = 8

# Collision detection (LiDAR based)
COLLISION_DIST  = 0.18  # entering collision zone threshold
COLLISION_CLEAR = 0.22  # leaving collision zone threshold

# ===== Arena → Grid mapping (same as your RL controller) =====
GRID_ROWS   = 12
GRID_COLS   = 12
ARENA_X_MIN = -1.35
ARENA_X_MAX =  1.35
ARENA_Y_MIN = -1.35
ARENA_Y_MAX =  1.35

CELL_SIZE_X = (ARENA_X_MAX - ARENA_X_MIN) / GRID_COLS
CELL_SIZE_Y = (ARENA_Y_MAX - ARENA_Y_MIN) / GRID_ROWS

# Coverage & obstacle mask
VISITED       = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
OBSTACLE_MASK = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

# Same obstacle cells as your RL controller (12x12 grid)
OBSTACLE_STATES = [
    26,27,28,31,32,
    38,39,40,43,44,
    62,63,64,65,66,67,68,69,
    74,75,76,77,78,79,80,81,
    86,87,88,89,90,91,92,93,
    116,117,
]

for s in OBSTACLE_STATES:
    r = s // GRID_COLS
    c = s %  GRID_COLS
    OBSTACLE_MASK[r, c] = True

# Number of free (non-obstacle) cells – coverage is relative to this
total_free_cells = int(np.count_nonzero(~OBSTACLE_MASK))

# ===================== STATE VARIABLES =====================
mode = "CRUISE"          # CRUISE (forward) / AVOID (obstacle avoidance)
omega_filt = 0.0
gap_err_filt = 0.0
omega_prev = 0.0
turn_sign_prev = 1.0
v_prev = 0.0
wander_omega = 0.0 

# experiment-related
sim_time      = 0.0
log_timer     = 0.0
collision_cnt = 0
in_collision  = False

x = y = 0.0
row_g = col_g = 0
coverage = 0.0

# ===================== UTILITY FUNCTIONS =====================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def lerp(a, b, t):
    return a + (b - a) * t

def sector_indices(n, fov, deg_from, deg_to):
    """
    Webots Lidar: index 0..n-1 corresponds to angles [-fov/2, +fov/2].
    Convert angle degrees into index range.
    """
    def ang_to_idx(deg):
        rad = math.radians(deg)
        frac = (rad + fov/2.0) / fov
        return int(clamp(round(frac * (n - 1)), 0, n - 1))
    i0 = ang_to_idx(deg_from)
    i1 = ang_to_idx(deg_to)
    return (min(i0, i1), max(i0, i1))

def min_in_sector(ranges, i0, i1):
    vals = [r for r in ranges[i0:i1+1] if math.isfinite(r)]
    return min(vals) if vals else 10.0

def avg_in_sector(ranges, i0, i1):
    vals = [r for r in ranges[i0:i1+1] if math.isfinite(r)]
    return sum(vals)/len(vals) if vals else 10.0

def find_widest_gap(ranges, safe):
    """
    Return (mid index, width) of widest safe gap.
    """
    n = len(ranges)
    best_len, best_mid = 0, n//2
    cur = None
    for i, r in enumerate(ranges):
        ok = math.isfinite(r) and r >= safe
        if ok and cur is None:
            cur = i
        elif not ok and cur is not None:
            L = i - cur
            if L > best_len:
                best_len = L
                best_mid = (cur + i - 1)//2
            cur = None
    if cur is not None:
        L = n - cur
        if L > best_len:
            best_len = L
            best_mid = (cur + n - 1)//2
    return best_mid, best_len

def pos_to_grid(x, y):
    """
    Map world coordinates (x, y) onto the 12x12 grid:
    (ARENA_X_MIN, ARENA_Y_MIN) → row=0, col=0 (bottom-left)
    (ARENA_X_MAX, ARENA_Y_MAX) → row=GRID_ROWS-1, col=GRID_COLS-1 (top-right)
    """
    col_f = (x - ARENA_X_MIN) / CELL_SIZE_X
    row_f = (y - ARENA_Y_MIN) / CELL_SIZE_Y

    col = int(clamp(math.floor(col_f), 0, GRID_COLS - 1))
    row = int(clamp(math.floor(row_f), 0, GRID_ROWS - 1))

    state_idx = row * GRID_COLS + col
    if DEBUG_PRINT:
        print(f"-> mapped row={row}, col={col}, cell_idx={state_idx}")
    return row, col, state_idx

def control_step(lidar_r, n, fov, v_prev, mode, omega_filt,
                 gap_err_filt, omega_prev, turn_sign_prev, dt):
    # ---- 1) Measurements ----
    iF0, iF1 = sector_indices(n, fov, -FRONT_DEG, +FRONT_DEG)
    d_front_min = min_in_sector(lidar_r, iF0, iF1)

    iL0, iL1 = sector_indices(n, fov, LF_DEG[0], LF_DEG[1])
    iR0, iR1 = sector_indices(n, fov, RF_DEG[0], RF_DEG[1])
    d_left_min  = min_in_sector(lidar_r, iL0, iL1)
    d_right_min = min_in_sector(lidar_r, iR0, iR1)
    d_left_avg  = avg_in_sector(lidar_r, iL0, iL1)
    d_right_avg = avg_in_sector(lidar_r, iR0, iR1)

    # Narrow front window check (to detect true "front obstacle")
    iN0, iN1 = sector_indices(n, fov, -NARROW_WIN_DEG, +NARROW_WIN_DEG)
    d_narrow = min_in_sector(lidar_r, iN0, iN1)

    v_now = max(v_prev, 0.01)
    ttc = d_narrow / v_now  # TTC based on narrow front window

    # ---- IMPORTANT CHANGE: we STOP using widest-gap / gap_err logic ----
    # No find_widest_gap, no gap_err, no gap_dir.

    # ---- 2) CRUISE / AVOID state machine (only safety-based) ----
    entering_avoid = (d_narrow < NARROW_ENTER) or (ttc < TTC_THRESH) or (d_front_min < D_SAFE)
    leaving_avoid  = (d_narrow > NARROW_LEAVE) and (ttc > TTC_THRESH + 0.6) and (d_front_min > D_SAFE + 0.15)

    if mode == "CRUISE" and entering_avoid:
        mode = "AVOID"
    elif mode == "AVOID" and leaving_avoid:
        mode = "CRUISE"

    # ---- 3) Wall-centering term only (no “go to open space”) ----
    # Right closer → turn left, and vice versa
    omega_wall = KP_WALL * (d_right_avg - d_left_avg)
    if INVERT_TURN_SIGN:
        omega_wall = -omega_wall

    # ---- 4) Speed cap from distance in front ----
    alpha = clamp((d_narrow - D_HARD) / max(1e-6, (D_SAFE - D_HARD)), 0.0, 1.0)
    v_cap = MIN_V + (MAX_V - MIN_V) * alpha

    # ---- 5) Desired v, ω  (NO gap steering) ----
    if mode == "CRUISE":
        # Just go forward with simple wall-centering – no preference for big gaps
        v_des = MAX_V
        omega_des = omega_wall + random.uniform(-0.1, 0.1)
        
    else:
        # AVOID mode: turn away from closer side, only for safety
        w_ttc = KP_TTC * max(0.0, TTC_THRESH - ttc)

        # If left side is closer (smaller distance), turn right, otherwise left
        turn_dir = 1.0 if (d_left_min < d_right_min) else -1.0
        if INVERT_TURN_SIGN:
            turn_dir = -turn_dir

        omega_des = turn_dir * w_ttc + omega_wall

        # If very narrow in front, bias even more strongly
        if d_narrow < 0.80:
            omega_des += turn_dir * 0.8

        # Slower in avoid mode
        v_des = max(MIN_V, MAX_V * 0.35)

        # Optional: small wall-following kick when facing a symmetric wall
        if EDGE_KICK_ENABLE:
            face_wall = (d_narrow < EDGE_FRONT_NEAR)
            sides_similar = abs(d_left_min - d_right_min) < EDGE_SIM_THRESH
            if face_wall and sides_similar:
                kick = (EDGE_KICK_W if EDGE_KEEP_RIGHT else -EDGE_KICK_W)
                if INVERT_TURN_SIGN:
                    kick = -kick
                omega_des += kick

    # ---- 6) Extremely close: force hard turn ----
    if d_narrow < D_HARD:
        v_des = MIN_V * 0.8
        hard_dir = 1.0 if (d_right_min > d_left_min) else -1.0
        if INVERT_TURN_SIGN:
            hard_dir = -hard_dir
        omega_des = hard_dir * (MAX_W * 0.95)

    # Enforce distance-based speed limit
    v_des = min(v_des, v_cap)

    # ---- 7) Smoothing & limits ----
    omega_des = clamp(omega_des, -MAX_W, +MAX_W)
    omega_smooth = SMOOTH * omega_filt + (1.0 - SMOOTH) * omega_des

    max_domega = OMEGA_SLEW * dt
    omega = omega_prev + clamp(omega_smooth - omega_prev, -max_domega, +max_domega)
    omega = clamp(omega, -MAX_W, +MAX_W)

    # Auto slow-down when turning sharply
    w_ratio = min(1.0, abs(omega) / MAX_W)
    v = lerp(v_des, V_AT_MAX_W, w_ratio)

    if DEBUG_PRINT:
        print(f"[{mode}] dN={d_narrow:.2f} dF={d_front_min:.2f} "
              f"dL={d_left_min:.2f} dR={d_right_min:.2f} "
              f"ttc={ttc:.2f} v_cap={v_cap:.2f} v={v:.2f} w={omega:.2f}")

    # gap_err_filt and turn_sign_prev are no longer used for steering,
    # but we keep them in the return signature to match the rest of your code.
    return v, omega, mode, omega, gap_err_filt, omega, turn_sign_prev


# Convert (v, ω) to wheel angular velocities
def v_omega_to_wheels(v, omega):
    wr = (v + 0.5 * omega * AXLE_LENGTH) / WHEEL_RADIUS
    wl = (v - 0.5 * omega * AXLE_LENGTH) / WHEEL_RADIUS
    return wr, wl

# ===================== Webots Initialization =====================
robot = Robot()
timestep = int(robot.getBasicTimeStep())
dt = max(1e-3, timestep / 1000.0)

# LiDAR
lidar = None
for name in ["LDS-01", "Hokuyo URG-04LX-UG01", "Sick LMS 291", "lidar"]:
    try:
        dev = robot.getDevice(name)
        if dev:
            lidar = dev
            break
    except Exception:
        pass
if lidar is None:
    raise RuntimeError("LiDAR not found (tried LDS-01 / Hokuyo / Sick / lidar)")
lidar.enable(timestep)
if hasattr(lidar, "enablePointCloud"):
    lidar.enablePointCloud()

# GPS (for coverage)
try:
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    HAS_GPS = True
except Exception:
    gps = None
    HAS_GPS = False
    print("[WARN] GPS not found, coverage will stay 0.")

# Motors
left  = robot.getDevice("left wheel motor")
right = robot.getDevice("right wheel motor")
for m in [left, right]:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Read max motor speed
try:
    MAX_WHEEL_RAD_S = min(
        left.getMaxVelocity() if hasattr(left, "getMaxVelocity") else 6.67,
        right.getMaxVelocity() if hasattr(right, "getMaxVelocity") else 6.67,
        8.0
    )
except Exception:
    MAX_WHEEL_RAD_S = 6.67

def apply_wheel_vel(wr, wl):
    wr = clamp(wr, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S)
    wl = clamp(wl, -MAX_WHEEL_RAD_S, MAX_WHEEL_RAD_S)
    if SWAP_WHEELS:
        left.setVelocity(wr)
        right.setVelocity(wl)
    else:
        right.setVelocity(wr)
        left.setVelocity(wl)

# ====== CSV log file ======
log_filename = f"log_{CONTROLLER_ID}_{WORLD_ID}_run{RUN_ID}.csv"
log_file = open(log_filename, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "time", "x", "y", "row", "col",
    "coverage", "collision_count", "v", "w"
])
print(f"[LOG] Writing to {log_filename}")

# ===================== Main Loop =====================
while robot.step(timestep) != -1:
    # --- time update ---
    sim_time += dt
    log_timer += dt

    ranges = lidar.getRangeImage()
    if not ranges:
        apply_wheel_vel(0.0, 0.0)
        continue

    # --- Collision counting (from LiDAR min distance) ---
    d_min = min([r for r in ranges if math.isfinite(r)] + [10.0])
    if d_min < COLLISION_DIST and not in_collision:
        collision_cnt += 1
        in_collision = True
    elif d_min > COLLISION_CLEAR and in_collision:
        in_collision = False

    n = lidar.getHorizontalResolution()
    fov = lidar.getFov()

    # --- Coverage update: use GPS position → grid cell ---
    if HAS_GPS:
        pos = gps.getValues()  # [x, y, z]; here we assume ground plane is (x, y)
        x, y = pos[0], pos[1]
        row_g, col_g, cell_idx = pos_to_grid(x, y)

        if 0 <= row_g < GRID_ROWS and 0 <= col_g < GRID_COLS:
            # Only count free cells as visited
            if not OBSTACLE_MASK[row_g, col_g]:
                VISITED[row_g, col_g] = True

        visited_free = np.count_nonzero(VISITED & (~OBSTACLE_MASK))
        if total_free_cells > 0:
            coverage = float(visited_free) / float(total_free_cells)
        else:
            coverage = 0.0
    else:
        x = y = 0.0
        row_g = col_g = 0
        coverage = 0.0

    # --- Baseline control ---
    v_cmd, w_cmd, mode, omega_filt, gap_err_filt, omega_prev, turn_sign_prev = control_step(
        ranges, n, fov, v_prev, mode, omega_filt, gap_err_filt, omega_prev, turn_sign_prev, dt
    )

    wr, wl = v_omega_to_wheels(v_cmd, w_cmd)
    apply_wheel_vel(wr, wl)
    v_prev = v_cmd

    if DEBUG_PRINT:
        print(f"[FINAL] t={sim_time:5.2f}s  row={row_g:2d}, col={col_g:2d}  "
              f"cov={coverage*100:5.1f}%  coll={collision_cnt}  v={v_cmd:.2f} w={w_cmd:.2f}")

    # ====== Logging to CSV ======
    if log_timer >= LOG_INTERVAL:
        log_writer.writerow([
            f"{sim_time:.2f}",
            f"{x:.3f}", f"{y:.3f}",
            row_g, col_g,
            f"{coverage:.3f}",
            collision_cnt,
            f"{v_cmd:.3f}", f"{w_cmd:.3f}"
        ])
        log_timer = 0.0

    # ====== End experiment after fixed duration ======
    if sim_time >= EXPERIMENT_DURATION:
        apply_wheel_vel(0.0, 0.0)
        print("\n==== BASELINE EXPERIMENT END ====")
        print(f"Time: {sim_time:.1f}s")
        print(f"Coverage (free cells only): {coverage*100:.1f}%")
        print(f"Collisions: {collision_cnt}")
        
        # Only count cells that are free (not obstacles)
        visited_indices = np.argwhere(np.logical_and(VISITED, ~OBSTACLE_MASK))

        print(f"\nTotal visited cells: {len(visited_indices)}/{total_free_cells}")
        print("Visited cell list (row, col):")
        for (r, c) in visited_indices:
            print(f"  ({r:2d}, {c:2d})")

        print("\nVisited grid (1 = visited, . = not):")
        for r in range(GRID_ROWS):
            line = ""
            for c in range(GRID_COLS):
                line += "1 " if VISITED[r, c] else ". "
            print(f"row {r:2d}: {line}")
            
        log_file.close()
        break
