from controller import Robot
from mapper import GridMapper
import matplotlib.pyplot as plt
import math
import numpy as np
import csv

# ==================== Mapping & Navigation Settings =====================

# Mapping
ENABLE_MAPPING = True
MAP_UPDATE_INTERVAL = 5  #Update map every 5 steps 
VIS_UPDATE_INTERVAL = 10  #Visualisation update interval

# Navigation maps 
MODE_RL_EXPLORATION = "RL_EXPLORATION" # RL exploration builds map
MODE_RETURNING_HOME = "RETURNING_HOME" # Navigating back home using A*
MODE_AT_HOME = "AT_HOME" # Arrived at home 

# Home position
HOME_X = 0.0
HOME_Y = 0.0


PHASE_1_DURATION = 60 
PHASE_2_DURATION = 240 
EXPLORATION_TIME = PHASE_1_DURATION + PHASE_2_DURATION


EXPERIMENT_DURATION = EXPLORATION_TIME

EDGE_STUCK_THRESHOLD = 3.0 
INTERVENTION_DURATION = 15.0   

# ====== Experiment settings  ======
EXPERIMENT_DURATION =EXPLORATION_TIME   # total experiment time (seconds)
LOG_INTERVAL        = 0.1     # record every 0.1 second
WORLD_ID            = "world1"
CONTROLLER_ID       = "rl_main_map"
RUN_ID              = 100

# Collision detection (LiDAR based)
COLLISION_DIST  = 0.18
COLLISION_CLEAR = 0.22


# Waypoint following parameters 
WAYPOINT_THRESHOLD = 0.12 # Distance to considered reached 

# ===================== RL Settings =====================

# Your .npy Q-table (relative to the Webots world path)
Q_TABLE_PATH = "../../rl/run226_qtable.npy"

# Action definitions (must match offline training: 0~4)
# 0=forward, 1=right, 2=left
ACTION_FORWARD  = 0
ACTION_RIGHT    = 1
ACTION_LEFT     = 2

# In Webots, we disable backward movement, so BACKWARD Q-values are suppressed
ACTION_DURATION = 1   # Duration of each discrete RL action (seconds)

# ===================== Arena → Grid Mapping =====================
# In this world the ground plane is (x, y); z is height.
# The world is roughly [-1.35, +1.35] × [-1.35, +1.35]

GRID_ROWS   = 12
GRID_COLS   = 12
ARENA_X_MIN = -1.35
ARENA_X_MAX =  1.35
ARENA_Y_MIN = -1.35
ARENA_Y_MAX =  1.35

CELL_SIZE_X = (ARENA_X_MAX - ARENA_X_MIN) / GRID_COLS
CELL_SIZE_Y = (ARENA_Y_MAX - ARENA_Y_MIN) / GRID_ROWS

# ===================== Coverage Tracking (optional) =====================

VISITED = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
OBSTACLE_MASK = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
coverage_last_print_time = -1.0

# These are the obstacle cells in the 12x12 grid (same as your other controller)
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


# ===================== Baseline Obstacle Avoidance Parameters =====================

# Front-distance / time thresholds
D_SAFE        = 0.25     # Safe distance: if front < D_SAFE → start AVOID
D_HARD        = 0.35     # Very close: force a hard turn
TTC_THRESH    = 1.2      # Time-To-Collision threshold (seconds)

# Emergency narrow front window
NARROW_WIN_DEG = 18
NARROW_ENTER   = 0.40    # <0.40 m → danger directly ahead
NARROW_LEAVE   = 0.55    # >0.55 m → exit AVOID mode

# Speed / gains
MAX_V = 1.2
MIN_V = 0.10
MAX_W = 3.0
KP_GAP  = 1.50       # (kept for future use, not used in steering now)
KP_TTC  = 3.20

# You still define KP_WALL etc., but they will NOT be used
KP_WALL = 0.90
EDGE_KICK_ENABLE = True
EDGE_KEEP_RIGHT  = True
EDGE_KICK_W      = 0.35
EDGE_SIM_THRESH  = 0.08
EDGE_FRONT_NEAR  = 0.85

# LiDAR sectors
FRONT_DEG   = 80
LF_DEG      = (0, +60)
RF_DEG      = (-60, 0)
GAP_SAFE    = 0.75
GAP_WIN_DEG = 70

# Stability parameters
SMOOTH         = 0.55
GAP_SMOOTH     = 0.55
GAP_DEAD_BAND  = 0.10
SIGN_STICK_EPS = 0.22
OMEGA_SLEW     = 3.5
V_AT_MAX_W     = 0.035

# RL influence strength on steering
RL_W_BIAS = 0.9   # Bias for RL left/right turn (rad/s)

# Quick toggles
INVERT_TURN_SIGN = False
SWAP_WHEELS = False

# Geometry
WHEEL_RADIUS = 0.033
AXLE_LENGTH  = 0.160

DEBUG_PRINT = True

# ===================== Path follwoing functions ================= 

def euclidean_distance(x1, y1, x2, y2):
    '''Calculate Euclidean distance between two points'''
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def go_to_waypoint(current_x, current_y, current_theta, goal_x, goal_y):
    '''
    Calculate velocities to navigate to a waypoint
    '''
    # Calculate distance and angle to goal 
    dx = goal_x - current_x
    dy = goal_y - current_y
    distance = math.sqrt(dx**2 + dy**2)
    
    # Calculate desired heading 
    desired_theta = math.atan2(dy, dx)
    
    # Calculate angle error (normalized to [-pi, pi])
    angle_error = desired_theta - current_theta
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
    
    # Check if waypoint reached
    if distance < WAYPOINT_THRESHOLD:
        return 0.0, 0.0, True
    
    # Proportional control
    K_LINEAR = 0.5
    K_ANGULAR = 2.0
    
    linear_vel = K_LINEAR * distance
    angular_vel = K_ANGULAR * angle_error
    
    # Limit velocities
    linear_vel = clamp(linear_vel, 0.0, MAX_V)
    angular_vel = clamp(angular_vel, -MAX_W, MAX_W)
    
    return linear_vel, angular_vel, False

def follow_path(current_x, current_y, current_theta, waypoints, current_waypoint_idx):
    if current_waypoint_idx >= len(waypoints):
        print("All waypoints completed.")
        return 0.0, 0.0, current_waypoint_idx, True
        
    print(f"Following waypoint {current_waypoint_idx}/{len(waypoints)}")    

    
    # Get current target waypoint
    target_x, target_y = waypoints[current_waypoint_idx]
    
    # Navigate to waypoint
    v, w, at_waypoint = go_to_waypoint(current_x, current_y, current_theta, target_x, target_y)
    
    if at_waypoint:
        #print(f" Reached waypoint {current_waypoint_idx + 1}/{len(waypoints)}")
        current_waypoint_idx += 1
        
        # Check if this was the last waypoint
        if current_waypoint_idx >= len(waypoints):
            return 0.0, 0.0, current_waypoint_idx, True
    
    return v, w, current_waypoint_idx, False

# ===================== Helper Functions =====================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def is_at_edge(row, col):
    return row == 0 or row == GRID_ROWS-1 or col == 0 or col == GRID_COLS-1

def get_inward_direction(row, col, orientation):
    center_row, center_col = GRID_ROWS // 2, GRID_COLS // 2
    
    if abs(row - center_row) <= 2 and abs(col - center_col) <= 2:
        return ACTION_FORWARD
    
    dr = center_row - row  
    dc = center_col - col  
    
    at_left = (col == 0)
    at_right = (col == GRID_COLS - 1)
    at_top = (row == GRID_ROWS - 1)
    at_bottom = (row == 0)
    
    if at_left and orientation == 1: 
        return ACTION_RIGHT 
    if at_right and orientation == 3:  
        return ACTION_LEFT 
    if at_bottom and orientation == 0:
        return ACTION_RIGHT 
    if at_top and orientation == 2: 
        return ACTION_LEFT 

    if orientation == 0: 
        if abs(dc) > abs(dr):
            return ACTION_RIGHT if dc > 0 else ACTION_LEFT
        else:
            return ACTION_FORWARD if dr > 0 else ACTION_LEFT
    elif orientation == 1: 
        if abs(dr) > abs(dc):
            return ACTION_RIGHT if dr < 0 else ACTION_LEFT
        else:
            return ACTION_FORWARD if dc > 0 else ACTION_LEFT
    elif orientation == 2: 
        if abs(dc) > abs(dr):
            return ACTION_LEFT if dc > 0 else ACTION_RIGHT
        else:
            return ACTION_FORWARD if dr < 0 else ACTION_RIGHT
    else: 
        if abs(dr) > abs(dc):
            return ACTION_LEFT if dr < 0 else ACTION_RIGHT
        else:
            return ACTION_FORWARD if dc < 0 else ACTION_RIGHT
            
            
        
        
def lerp(a, b, t):
    return a + (b - a) * t

def sector_indices(n, fov, deg_from, deg_to):
    """
    Webots Lidar: index 0..n−1 corresponds to angles [-fov/2, +fov/2].
    Convert angle degrees into an index range.
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
    Return (mid index, width) of the widest safe gap.
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

# ===================== Grid Mapping =====================
def pos_to_grid(x, y):
    """
    (ARENA_X_MIN, ARENA_Y_MIN) → (row=0, col=0)          # bottom-left
    (ARENA_X_MAX, ARENA_Y_MAX) → (row=GRID_ROWS-1, col=GRID_COLS-1)  # top-right
    """
    col_f = (x - ARENA_X_MIN) / CELL_SIZE_X
    row_f = (y - ARENA_Y_MIN) / CELL_SIZE_Y

    col = int(clamp(math.floor(col_f), 0, GRID_COLS - 1))
    row = int(clamp(math.floor(row_f), 0, GRID_ROWS - 1))

    state_idx = row * GRID_COLS + col
    #print(f"-> mapped row={row}, col={col}, cell_idx={state_idx}")
    return row, col, state_idx
    
def get_orientation_index():
     #0 = North, 1 = East, 2 = South, 3 = West,Compass Orientation
    north = compass.getValues()
    heading = math.atan2(-north[0], north[1])  # x, z → yaw

    if heading < 0:
        heading += 2 * math.pi

    if heading < math.pi/4 or heading >= 7*math.pi/4:
        return 0  # North
    elif heading < 3*math.pi/4:
        return 1  # East
    elif heading < 5*math.pi/4:
        return 2  # South
    else:
        return 3  # West


# ===================== Q-table Loading & Action Selection =====================

def load_q_table(path):
    Q = np.load(path)
    return Q

def select_rl_action(Q, state_idx):
    """
    Pure greedy Q-table action selection.
    - BACKWARD is always disabled.
    No LiDAR-based safety mask here: RL has full control.
    """
    row = Q[state_idx].copy()

    # Argmax over remaining actions (forward / right / left)
    return int(np.argmax(row))

# ===================== Baseline + RL Control Step =====================

def control_step(lidar_r, n, fov,
                 v_prev, mode, omega_filt, gap_err_filt, omega_prev, turn_sign_prev,
                 dt, rl_action, phase=1, intervention_mode=False, debug=False):

    # ---- 1) LiDAR measurements ----
    d_safe_use = D_SAFE if phase == 1 else D_SAFE * 0.85  # Phase 2 降低15%
    d_hard_use = D_HARD if phase == 1 else D_HARD * 0.90
    
    iF0, iF1 = sector_indices(n, fov, -FRONT_DEG, +FRONT_DEG)
    d_front_min = min_in_sector(lidar_r, iF0, iF1)

    iL0, iL1 = sector_indices(n, fov, LF_DEG[0], LF_DEG[1])
    iR0, iR1 = sector_indices(n, fov, RF_DEG[0], RF_DEG[1])
    d_left_min  = min_in_sector(lidar_r, iL0, iL1)
    d_right_min = min_in_sector(lidar_r, iR0, iR1)
    d_left_avg  = avg_in_sector(lidar_r, iL0, iL1)
    d_right_avg = avg_in_sector(lidar_r, iR0, iR1)

    # Narrow forward window
    iN0, iN1 = sector_indices(n, fov, -NARROW_WIN_DEG, +NARROW_WIN_DEG)
    d_narrow = min_in_sector(lidar_r, iN0, iN1)

    v_now = max(v_prev, 0.01)
    ttc = d_narrow / v_now

    # ---- 2) Find widest gap (within ±GAP_WIN_DEG) ----
    g0, g1 = sector_indices(n, fov, -GAP_WIN_DEG, +GAP_WIN_DEG)
    sub = lidar_r[g0:g1+1]
    mid_local, _ = find_widest_gap(sub, GAP_SAFE)
    mid_idx = g0 + mid_local

    # Convert to [-1, +1] error (right positive, left negative)
    gap_err_raw = (mid_idx - (n - 1) / 2.0) / (n / 2.0)
    if INVERT_TURN_SIGN:
        gap_err_raw = -gap_err_raw

    # Low-pass + deadband
    gap_err_filt = GAP_SMOOTH * gap_err_filt + (1.0 - GAP_SMOOTH) * gap_err_raw
    gap_err = 0.0 if abs(gap_err_filt) < GAP_DEAD_BAND else gap_err_filt

    # Sign-sticking (not used directly in steering anymore, but kept for debug)
    if abs(gap_err) < SIGN_STICK_EPS:
        turn_sign = 1.0 if omega_prev >= 0.0 else -1.0
        if abs(omega_prev) < 1e-3:
            turn_sign = turn_sign_prev
        gap_dir = turn_sign
    else:
        gap_dir = 1.0 if gap_err >= 0 else -1.0
    turn_sign_prev = gap_dir

    # --- RL steering bias from discrete action ---
    omega_rl = 0.0
    v_rl_extra = 0.0  # no STOP effect anymore

    if rl_action == ACTION_LEFT:
        omega_rl = +RL_W_BIAS
    elif rl_action == ACTION_RIGHT:
        omega_rl = -RL_W_BIAS
    # ACTION_STOP is effectively removed: no branch for it

    if INVERT_TURN_SIGN:
        omega_rl = -omega_rl


    # ---- 3) CRUISE / AVOID state machine ----
    entering_avoid = (
        d_narrow < NARROW_ENTER or
        ttc < TTC_THRESH or
        d_front_min < d_safe_use
    )
    leaving_avoid = (
        d_narrow > NARROW_LEAVE and
        ttc > TTC_THRESH + 0.5 and
        d_front_min > d_safe_use + 0.10
    )

    if mode == "CRUISE" and entering_avoid:
        mode = "AVOID"
    elif mode == "AVOID" and leaving_avoid:
        mode = "CRUISE"

    # ---- 4) Limit max speed based on distance ----
    alpha = clamp((d_narrow - d_hard_use) / max(1e-6, (d_safe_use - d_hard_use)), 0.0, 1.0)
    v_cap = MIN_V + (MAX_V - MIN_V) * alpha
    

    # ---- 5) Desired v, ω ----
    if mode == "CRUISE":
        # *** RL ONLY *** (no wall-following, no gap-centering)
        v_des = MAX_V + v_rl_extra
        v_des = max(MIN_V, min(MAX_V, v_des))
        omega_des = omega_rl        
        if intervention_mode:
            v_des = MAX_V * 0.8 
            if rl_action == ACTION_LEFT:
                omega_des = +2.0 
            elif rl_action == ACTION_RIGHT:
                omega_des = -2.0
            else:
                omega_des = 0.0           
    else:
        # AVOID mode: safety via TTC, RL decides direction
        w_ttc = KP_TTC * max(0.0, TTC_THRESH - ttc)
        # Side with more space (used when RL has no turn preference)
        prefer_right = (d_right_min > d_left_min)
        hard_dir = 1.0 if prefer_right else -1.0

        # Direction preference comes from RL if it is a turning action
        if rl_action == ACTION_LEFT:
            avoid_dir = +1.0
        elif rl_action == ACTION_RIGHT:
            avoid_dir = -1.0
        else:
            avoid_dir = hard_dir  # if RL says forward/stop, fall back to safer side

        if INVERT_TURN_SIGN:
            hard_dir  = -hard_dir
            avoid_dir = -avoid_dir

        # RL bias + TTC-based turning in RL's preferred direction
        omega_des = omega_rl + avoid_dir * w_ttc

        # If very narrow, turn harder (still in avoid_dir)
        if d_narrow < 0.80:
            omega_des += avoid_dir * 0.8

        v_des = max(MIN_V, MAX_V * 0.45 + v_rl_extra)

    # ---- 6) Extremely close → force hard turn, no reversing ----
    if d_narrow < d_hard_use:
        v_des = MIN_V * 0.8
        hard_dir = 1.0 if (d_right_min > d_left_min) else -1.0
        if INVERT_TURN_SIGN:
            hard_dir = -hard_dir
        omega_des = hard_dir * (MAX_W * 0.95)
    

    # Never allow v < 0 (no backward)
    v_des = max(0.0, v_des)
    v_des = min(v_des, v_cap)

    # ---- 7) Smoothing & limiting ----
    omega_des = clamp(omega_des, -MAX_W, +MAX_W)
    omega_smooth = SMOOTH * omega_filt + (1.0 - SMOOTH) * omega_des

    max_domega = OMEGA_SLEW * dt
    omega = omega_prev + clamp(omega_smooth - omega_prev, -max_domega, +max_domega)
    omega = clamp(omega, -MAX_W, +MAX_W)

    # Auto slow-down while turning
    w_ratio = min(1.0, abs(omega) / MAX_W)
    v = lerp(v_des, V_AT_MAX_W, w_ratio)
    
    if debug:
        mode_tag = "CRUISE" if mode == "CRUISE" else "AVOID"
        print(f"[{mode_tag}] dN={d_narrow:.2f} dF={d_front_min:.2f} "
              f"dL={d_left_min:.2f} dR={d_right_min:.2f} ttc={ttc:.2f} "
              f"v_cap={v_cap:.2f} v={v:.2f} w={omega:.2f}")
   
    return v, omega, mode, omega_smooth, gap_err_filt, omega, turn_sign_prev

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

# GPS (world must include a GPS named "gps")
try:
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    HAS_GPS = True
except Exception:
    gps = None
    HAS_GPS = False
    #print("[WARN] GPS not found, RL state will always be 0.")

# Add compass for orientation
compass = robot.getDevice("compass")
compass.enable(timestep)

# Motors
left  = robot.getDevice("left wheel motor")
right = robot.getDevice("right wheel motor")
for m in [left, right]:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Motor speed limit
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

# ===================== Load Q-table =====================

Q_table = load_q_table(Q_TABLE_PATH)
NUM_STATES, NUM_ACTIONS = Q_table.shape
print(f"Loaded Q-table: {Q_table.shape}")

# ===================== Initialise mapper ===================
if ENABLE_MAPPING:
    # Create mapper (adjust size to your arena: ~2.7m x 2.7m)
    mapper = GridMapper(map_size_meters=3.0, resolution=0.05)
    
    # Setup matplotlib visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    map_display = ax.imshow(
        mapper.get_map_image(),
        cmap='gray',
        origin='lower',
        vmin=0,
        vmax=1
    )
    ax.set_title('Real-time Occupancy Grid Map (RL Agent)')
    ax.set_xlabel('Grid X')
    ax.set_ylabel('Grid Y')
    robot_marker, = ax.plot([], [], 'ro', markersize=8, label='Robot')
    path_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5, label='Path')
    planned_path_line, = ax.plot([], [], 'g-', linewidth=2, alpha=0.7, label='A* Path')
    ax.legend()
    plt.colorbar(map_display, ax=ax, label='Occupancy (0=free, 1=occupied)')
    
    #print("Mapping enabled: Building occupancy grid...")
else:
    mapper = None
    #print("Mapping disabled.")

# ====== Experiment log file ======
log_filename = f"log_{CONTROLLER_ID}_{WORLD_ID}_run{RUN_ID}.csv"
log_file = open(log_filename, "w", newline="")
log_writer = csv.writer(log_file)
log_writer.writerow([
    "time", "x", "y", "row", "col",
    "coverage", "collision_count", "v", "w"
])
print(f"[LOG] Writing to {log_filename}")

# ===================== State Variables =====================

mode = "CRUISE"
omega_filt = 0.0
gap_err_filt = 0.0
omega_prev = 0.0
turn_sign_prev = 1.0
v_prev = 0.0

# RL-related
rl_action = ACTION_FORWARD
action_timer = 0.0
last_state_idx = 0
last_row = 0
last_col = 0

# Navigation state
nav_mode = MODE_RL_EXPLORATION
current_phase = 1
phase_switch_time = None 
edge_time_accumulator = 0.0 
last_was_edge = False
intervention_active = False 
intervention_timer = 0.0 
intervention_direction = None 
exploration_start_time = None 
planned_waypoints = []
current_waypoints_idx = 0 

# Counters 
step_count = 0 
map_update_count = 0

# ====== Coverage / experiment variables ======
total_free_cells = int(np.count_nonzero(~OBSTACLE_MASK))

x = y = 0.0
row_g = col_g = 0
coverage = 0.0

sim_time      = 0.0
log_timer     = 0.0
collision_cnt = 0
in_collision  = False

# ===================== Main Loop =====================

print(f"\n{'='*60}")
print(f"Starting RL Agent with mapping")
print(f"Exploration time: {EXPLORATION_TIME}s")
print(f"Home positon: ({HOME_X:.2f}, {HOME_Y:.2f})")
print(f"{'='*60}\n")

while robot.step(timestep) != -1:
    # --- time update ---
    sim_time += dt
    log_timer += dt

    ranges = lidar.getRangeImage()
    if not ranges:
        apply_wheel_vel(0.0, 0.0)
        continue

    # --- collision counting (from LiDAR min distance) ---
    d_min = min([r for r in ranges if math.isfinite(r)] + [10.0])
    if d_min < COLLISION_DIST and not in_collision:
        collision_cnt += 1
        in_collision = True
    elif d_min > COLLISION_CLEAR and in_collision:
        in_collision = False


    n = lidar.getHorizontalResolution()
    fov = lidar.getFov()
    
    # Get robot position and orientation 
    if HAS_GPS:
        pos = gps.getValues()
        x, y = pos[0], pos[1]
        #print(f"GPS x={x:+.3f}, y={y:+.3f}")
        row, col, state_idx = pos_to_grid(x, y)
        #print(f"→ row={row}, col={col}, cell_idx={state_idx}")
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            if not OBSTACLE_MASK[row, col]:
                VISITED[row, col] = True
        # Get orientation from compass 
        north = compass.getValues()
        theta = math.atan2(north[0], north[2])
    else:
        x = y = theta = 0.0
        #print("[ERROR] GPS not available - mapping requires GPS!")
        
    # Update Map
    if ENABLE_MAPPING and HAS_GPS and (step_count % MAP_UPDATE_INTERVAL == 0):
        lidar_angles = np.linspace(-fov/2, fov/2, n)
        mapper.update_map(x, y, theta, ranges, lidar_angles)
        map_update_count += 1 
        
    # Visualisation 
    if ENABLE_MAPPING and (step_count % VIS_UPDATE_INTERVAL == 0):
        # Update map display
        map_display.set_data(mapper.get_map_image())
        
        # Update robot position on map
        robot_grid_x, robot_grid_y = mapper.world_to_grid(x, y)
        robot_marker.set_data([robot_grid_x], [robot_grid_y])
        
        # Update path trail
        if len(mapper.robot_path) > 0:
            path_x, path_y = zip(*mapper.robot_path)
            path_line.set_data(path_x, path_y)
        
        # Update planned path if available
        if len(planned_waypoints) > 0:
            waypoints_grid = [mapper.world_to_grid(wx, wy) for wx, wy in planned_waypoints]
            if len(waypoints_grid) > 0:
                wx, wy = zip(*waypoints_grid)
                planned_path_line.set_data(wx, wy)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    # =============== Navigation State Machine ====================
    
    if nav_mode == MODE_RL_EXPLORATION:
        # Initialise timer on first iteration 
        if exploration_start_time is None:
            exploration_start_time = robot.getTime()
            
        # Check if exploration time is up 
        elapsed_time = robot.getTime() - exploration_start_time
        
        if current_phase == 1 and elapsed_time >= PHASE_1_DURATION:
            current_phase = 2
            phase_switch_time = robot.getTime()        
        
        if elapsed_time >= EXPLORATION_TIME:
            print(f"\n{'='*60}")
            print(f"Exploration complete! ({map_update_count} map updates)")
            print(f"Planning path home using A* on built map.....")
            print(f"{'='*60}\n")
           
            # Path home 
            if ENABLE_MAPPING:
                planned_waypoints = mapper.plan_path_astar(x, y, HOME_X, HOME_Y)
                print(f"Planned waypoints: {planned_waypoints}")

                if len(planned_waypoints) > 0:
                    #print(f" Path planned with {len(planned_waypoints)} waypoints")
                    current_waypoint_idx = 0
                    nav_mode = MODE_RETURNING_HOME
                    print("Switched to RETURNING_HOME mode.")

                else:
                    #print("✗ Could not plan path! Stopping.")
                    nav_mode = MODE_AT_HOME
            else:
                #print("Mapping disabled, cannot plan path.")
                nav_mode = MODE_AT_HOME

        
    # --- Safety analysis for RL actions (simple LiDAR checks) ---
    iF0, iF1 = sector_indices(n, fov, -FRONT_DEG, +FRONT_DEG)
    d_front_min = min_in_sector(ranges, iF0, iF1)

    iL0, iL1 = sector_indices(n, fov, LF_DEG[0], LF_DEG[1])
    iR0, iR1 = sector_indices(n, fov, RF_DEG[0], RF_DEG[1])
    d_left_min  = min_in_sector(ranges, iL0, iL1)
    d_right_min = min_in_sector(ranges, iR0, iR1)
    
    if nav_mode == MODE_RETURNING_HOME and ENABLE_MAPPING and HAS_GPS:
       pos = gps.getValues()
       x, y = pos[0], pos[1]
       north = compass.getValues()
       theta = math.atan2(north[0], north[2])
       print(f"Executing RETURNING_HOME: waypoint {current_waypoint_idx}")

       v_pf, w_pf, current_waypoint_idx, done = follow_path(x, y, theta, planned_waypoints, current_waypoint_idx)
    
       if done:
           print("back to origin point.")
           nav_mode = MODE_AT_HOME
           v_pf, w_pf = 0.0, 0.0

       wr, wl = v_omega_to_wheels(v_pf, w_pf)
       apply_wheel_vel(wr, wl)
       v_prev = v_pf

       continue 


    if current_phase == 2 and HAS_GPS:
        pos = gps.getValues()
        x_check, y_check = pos[0], pos[1]
        row_check, col_check, _ = pos_to_grid(x_check, y_check)
        
        is_edge = is_at_edge(row_check, col_check)
        
        if is_edge:
            edge_time_accumulator += dt  # 每个timestep累积
            if edge_time_accumulator >= EDGE_STUCK_THRESHOLD and not intervention_active:
                orientation_check = get_orientation_index()                
                intervention_active = True
                intervention_timer = INTERVENTION_DURATION
                intervention_direction = get_inward_direction(row_check, col_check, orientation_check) 
                edge_time_accumulator = 0.0
        else:
            if last_was_edge:
                edge_time_accumulator = max(0.0, edge_time_accumulator - dt * 0.5)
        
        last_was_edge = is_edge
    # ---------- RL state update & action selection ----------
    debug_new_state = False
    if action_timer <= 0.0:
        if HAS_GPS:
            pos = gps.getValues()  # [x, y, z]
            x, y = pos[0], pos[1]
            row_g, col_g, cell_idx = pos_to_grid(x, y)
            # --- Coverage update ---
            if 0 <= row_g < GRID_ROWS and 0 <= col_g < GRID_COLS:
                if not OBSTACLE_MASK[row_g, col_g]:
                    VISITED[row_g, col_g] = True

            visited_free = np.count_nonzero(VISITED & (~OBSTACLE_MASK))
            if total_free_cells > 0:
                coverage = float(visited_free) / float(total_free_cells)
            else:
                coverage = 0.0
                                               
            orientation = get_orientation_index()    
            s_idx = cell_idx * 4 + orientation
        else:
            x = y = 0.0
            row_g = col_g = 0
            orientation = 0
            s_idx = 0

        s_idx = int(clamp(s_idx, 0, NUM_STATES - 1))
        debug_new_state = (s_idx != last_state_idx)
        
        if debug_new_state and HAS_GPS:
          
            print(f"GPS x={x:+.3f}, y={y:+.3f}")
            print(f"-> mapped row={row_g}, col={col_g}, cell_idx={cell_idx}")
            print(f"→ row={row_g}, col={col_g}, cell_idx={cell_idx}")

        last_state_idx = s_idx
        last_row, last_col = row_g, col_g

        if intervention_active:
            rl_action = intervention_direction           
        else:
            rl_action = select_rl_action(Q_table, s_idx)          
            
        if debug_new_state and HAS_GPS:
            print(f"RL state: x={x:+.2f}, y={y:+.2f} "
                  f"-> row={row_g}, col={col_g}, orientation={orientation}, "
                  f"s={s_idx}, act={rl_action}")                     

        action_timer = ACTION_DURATION
       
    # Decrease remaining duration for current RL action
    action_timer -= dt
        # ---------- Combined Control ----------
    v_cmd, w_cmd, mode, omega_filt, gap_err_filt, omega_prev, turn_sign_prev = control_step(
        ranges, n, fov,
        v_prev, mode, omega_filt, gap_err_filt, omega_prev, turn_sign_prev,
        dt, rl_action, phase=current_phase,
        intervention_mode=intervention_active,
        debug=debug_new_state  
    )

    # ---------- Apply Wheel Speeds ----------
    wr, wl = v_omega_to_wheels(v_cmd, w_cmd)
    apply_wheel_vel(wr, wl)
    v_prev = v_cmd
    
    if debug_new_state and HAS_GPS:
        print(f"[FINAL] s={last_state_idx:3d} "
              f"(r={last_row:2d},c={last_col:2d}) "
              f"RL_act={rl_action} v={v_cmd:.2f} w={w_cmd:.2f}")
    
    # increment step counter
    step_count += 1
    
    if intervention_active:
        intervention_timer -= dt
    if intervention_timer <= 0.0:
        intervention_active = False
       
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
        
     # ====== Stop experiment after fixed duration ======
    if sim_time >= EXPERIMENT_DURATION:
        apply_wheel_vel(0.0, 0.0)
        print("\n==== RL + Mapping EXPERIMENT END ====")
        print(f"Time: {sim_time:.1f}s")
        print(f"Coverage: {coverage*100:.1f}%")
        print(f"Collisions: {collision_cnt}")

        # Debug: visited cells
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
