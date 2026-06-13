import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from dynamixel_sdk import *
from ball_velocity_tracker import track_ping_pong_ball

# ─── HARDWARE & CALIBRATION SETUP ─────────────────────────────────────────
DXL_IDS          = [3, 2, 1]
BAUDRATE         = 57600
PROTOCOL_VERSION = 2.0
# DEVICENAME       = "COM3" # Windows
DEVICENAME       = "/dev/tty.usbserial-FT3FSNI8"

ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION    = 4
POSITION_CONTROL_MODE = 3

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION  = 4

ball_time_history = []
error_x_history   = []
error_y_history   = []

cap = None

# --- GEOMETRY SETTINGS (Your updated compact dimensions) ---
R_BASE  = 87.5      
R_PLATE = 129.5     
L_ARM   = 42.0      
L_ROD   = 64.0      

ANGLES  = np.radians([0, 120, 240])
B       = np.array([R_BASE * np.cos(ANGLES), R_BASE * np.sin(ANGLES), np.zeros(3)])
P_LOCAL = np.array([R_PLATE * np.cos(ANGLES), R_PLATE * np.sin(ANGLES), np.zeros(3)])
BETA    = ANGLES

# Hardware Calibration Limits
Q_MAX_HORIZONTAL = np.array([188.0, 162.0, 172.0]) 
Q_MIN_VERTICAL   = np.array([98.0, 72.0, 82.0])    

# --- NEUTRAL BALANCING PLANE ---
NEUTRAL_Z = 80.0  # Safe compact mid-height for your short rods

# ─── PID CONTROLLER TUNING PARAMETERS ──────────────────────────────────────
K_P_ROLL  = 8    # Proportional gain for X -> Roll tilt
K_D_ROLL  = 4    # Derivative gain (damping) for X speed
K_I_ROLL  = 2    # Integral gain for Roll

K_P_PITCH = K_P_ROLL    # Proportional gain for Y -> Pitch tilt
K_D_PITCH = K_D_ROLL    # Derivative gain (damping) for Y speed
K_I_PITCH = K_I_ROLL    # Integral gain for Pitch


MAX_TILT_DEG = 8.0  # Hard soft-stop cap to keep angles gentle and stable

# ─── DATA LOGGING / CSV OUTPUT SETTINGS ────────────────────────────────────
# Change this path to control where the CSV results are saved.
CSV_OUTPUT_DIR = "/Users/juliachen/Desktop/plotz/variablestarting"

# Will be set at runtime based on user input (e.g. "pingpong" or "golf")
BALL_TYPE = "ball"

# ─── INTERACTION HELPER FUNCTIONS ──────────────────────────────────────────────────
def deg_to_pos_ticks(deg):
    return int(deg * (4096.0 / 360.0))

def pos_ticks_to_deg(ticks):
    return ticks * (360.0 / 4096.0)

def _signed_deg_err(current, target):
    err = (target - current + 180) % 360 - 180
    return err

def prompt_for_ball_type():
    """Asks the user (via the command window) what type of ball is being used.
    This is used later to label the saved CSV file."""
    global BALL_TYPE
    while True:
        response = input("Enter ball type ('ping pong' or 'golf'): ").strip().lower()
        if response in ("ping pong", "pingpong", "ping-pong", "pp"):
            BALL_TYPE = "ping_pong"
            break
        elif response in ("golf", "golf ball", "golfball"):
            BALL_TYPE = "golf"
            break
        else:
            print("Invalid entry. Please type 'ping pong' or 'golf'.")
    print(f"Ball type set to: {BALL_TYPE}")


def get_ball_position(camera_index=0):
    """
    Masks out the white background by isolating the black circular platform,
    then securely tracks the white ping-pong ball moving inside it.
    """
    global cap
    import cv2
    import numpy as np
    
    
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CENTER_X = int(FRAME_WIDTH / 2)   # 320
    CENTER_Y = int(FRAME_HEIGHT / 2)  # 240

    if cap is None:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            return 0.0, 0.0

    ret, frame = cap.read()
    if not ret or frame is None:
        return 0.0, 0.0

    # --- CONVERT TO GRAYSCALE & BLUR ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # --- CREATE A BLACK MASK FOR THE BACKGROUND ---
    bg_mask = np.zeros_like(gray)
    
    # Use Hough Circles to find platform plate
    platform_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=150, maxRadius=240
    )

    # Default platform coordinates if Hough Circles misses it on a frame
    plat_x, plat_y, plat_r = CENTER_X, CENTER_Y, 200 

    if platform_circles is not None:
        platform_circles = np.uint16(np.around(platform_circles))
        # Take the largest detected circle (the platform)
        plat_x, plat_y, plat_r = platform_circles[0][0]

    # Draw a pure white circle on our black canvas mask to serve as the "window"
    cv2.circle(bg_mask, (plat_x, plat_y), plat_r, 255, -1)

    # --- MASK THE GRAYSCALE IMAGE ---
    # Copy pixels ONLY where bg_mask is white. Everything else becomes black (0).
    masked_gray = cv2.bitwise_and(gray, gray, mask=bg_mask)

    # --- TRACK THE WHITE BALL INSIDE THE CLEAN ZONE ---
    _, thresh = cv2.threshold(masked_gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_ball_pixel_x = None
    detected_ball_pixel_y = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Keep our filters focused on the ball's expected pixel scale
        if area < 250: 
            continue
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        
        if circularity > 0.45: 
            if area > max_area:
                max_area = area
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    detected_ball_pixel_x = float(M["m10"] / M["m00"])
                    detected_ball_pixel_y = float(M["m01"] / M["m00"])

    # --- DRAW UI ELEMENTS ON THE ORIGINAL DISPLAY FRAME ---
    # Red crosshair for calibration alignment
    cv2.circle(frame, (CENTER_X, CENTER_Y), 4, (0, 0, 255), -1)
    # Draw a thin yellow line showing where the background mask boundary cuts off
    cv2.circle(frame, (plat_x, plat_y), plat_r, (0, 255, 255), 1)

    if detected_ball_pixel_x is not None and detected_ball_pixel_y is not None:
        # Normalize coordinates relative to the frame center
        centered_x = detected_ball_pixel_x - CENTER_X
        centered_y = CENTER_Y - detected_ball_pixel_y  

        norm_x = centered_x / CENTER_X
        norm_y = centered_y / CENTER_Y

        # Render a green target indicator around the ball
        cv2.circle(frame, (int(detected_ball_pixel_x), int(detected_ball_pixel_y)), 18, (0, 255, 0), 2)
        cv2.putText(frame, f"Ball: {norm_x:.2f}, {norm_y:.2f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        
        cv2.imshow("Stewart Platform Tracking Feed", frame)
        cv2.imshow("Thresholded Mask View", thresh)
        cv2.waitKey(1) 
        return norm_x, norm_y
    else:
        cv2.putText(frame, "BALL LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Stewart Platform Tracking Feed", frame)
        cv2.imshow("Thresholded Mask View", thresh)
        cv2.waitKey(1)
        return 0.0, 0.0


def inverse_kinematics(roll_deg, pitch_deg, z_mm):
    """Computes exact motor angles required for a target spatial plate pose."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)

    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    R = Rx @ Ry

    P_global = np.array([[0], [0], [z_mm]]) + R @ P_LOCAL
    motor_angles_physical = np.zeros(3)

    for i in range(3):
        V = P_global[:, i] - B[:, i]
        x_proj = V[0] * np.cos(BETA[i]) + V[1] * np.sin(BETA[i])
        z_proj = V[2]

        rho_sq = x_proj**2 + z_proj**2
        val_cos = (rho_sq + L_ARM**2 - L_ROD**2) / (2 * L_ARM * np.sqrt(rho_sq))

        if abs(val_cos) > 1.0:
            return None, False

        theta_math = np.arctan2(z_proj, x_proj) - np.arccos(val_cos)
        motor_angles_physical[i] = Q_MAX_HORIZONTAL[i] - np.degrees(theta_math)

        if (motor_angles_physical[i] < Q_MIN_VERTICAL[i]) or (motor_angles_physical[i] > Q_MAX_HORIZONTAL[i]):
            return None, False

    return motor_angles_physical, True


def goto_home_via_position_mode(
    q_home_deg,
    packetHandler=None,
    portHandler=None,
    pos_tol_deg: float = 1.0,
    timeout_s: float = 5.0,
    profile_velocity: int = 60,
):
    """Drives platform smoothly to a safe initial state using standard blocking position calls."""
    if packetHandler is None or portHandler is None:
        raise ValueError("packetHandler and portHandler must be provided.")
        
    current_deg = []
    for dxl_id in DXL_IDS:
        pos_raw, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
        current_deg.append(pos_ticks_to_deg(pos_raw))

    print(f"Homing... (current pose: {[f'{a:.1f}°' for a in current_deg]} -> target: {q_home_deg})")

    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE)
        packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_PROFILE_VELOCITY, profile_velocity)
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

    for dxl_id, ang_deg in zip(DXL_IDS, q_home_deg, strict=True):
        packetHandler.write4ByteTxRx(portHandler, dxl_id, ADDR_GOAL_POSITION, deg_to_pos_ticks(ang_deg))

    t0 = time.time()
    while True:
        if time.time() - t0 > timeout_s:
            print("Notice: Homing reached timeout window boundary.")
            break
        all_ok = True
        for dxl_id, ang_deg in zip(DXL_IDS, q_home_deg, strict=True):
            pos_raw, _, _ = packetHandler.read4ByteTxRx(portHandler, dxl_id, ADDR_PRESENT_POSITION)
            if abs(_signed_deg_err(pos_ticks_to_deg(pos_raw), ang_deg)) > pos_tol_deg:
                all_ok = False
                break
        if all_ok:
            break
        time.sleep(0.05)
    print("Homed and ready.")


def _build_csv_filename():
    """Builds a CSV filename that encodes the PID gains and ball type."""
    filename = (
        f"P{K_P_ROLL}_I{K_I_ROLL}_D{K_D_ROLL}_{BALL_TYPE}.csv"
    )
    return filename


def save_data_to_csv():
    """Saves timestamp and linear distance-from-center data to a CSV file
    for later plotting/comparison. The file is named using the PID gains
    and the ball type, and saved to CSV_OUTPUT_DIR."""
    if not ball_time_history:
        print("No dynamic data collected during execution. Skipping CSV save.")
        return

    # Scale: normalized [-1,1] -> pixels (×320) -> mm (×535/640) -> cm (÷10)
    PX_PER_NORM = 320
    MM_PER_PX   = 535.0 / 640.0
    CM_PER_NORM = PX_PER_NORM * MM_PER_PX / 10.0   # = 26.75 cm per unit

    error_x_cm = np.array(error_x_history, dtype=float) * CM_PER_NORM
    error_y_cm = np.array(error_y_history, dtype=float) * CM_PER_NORM
    distance_cm = np.sqrt(error_x_cm**2 + error_y_cm**2)

    # Make sure the output directory exists
    os.makedirs(CSV_OUTPUT_DIR, exist_ok=True)

    filename = _build_csv_filename()
    filepath = os.path.join(CSV_OUTPUT_DIR, filename)

    with open(filepath, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_s", "distance_cm", "error_x_cm", "error_y_cm"])
        for t, d, ex, ey in zip(ball_time_history, distance_cm, error_x_cm, error_y_cm):
            writer.writerow([f"{t:.6f}", f"{d:.6f}", f"{ex:.6f}", f"{ey:.6f}"])

    print(f"Saved data log to: {filepath}")


def plot_ball_error_performance():
    if not ball_time_history:
        print("No dynamic data collected during execution. Skipping plots.")
        return

    # Scale: normalized [-1,1] -> pixels (×320) -> mm (×535/640) -> cm (÷10)
    PX_PER_NORM = 320                  # 1.0 normalized = 320 px (half frame width)
    MM_PER_PX   = 535.0 / 640.0       # physical calibration
    CM_PER_NORM = PX_PER_NORM * MM_PER_PX / 10.0   # = 26.75 cm per unit
    error_x_cm = np.array(error_x_history, dtype=float) * CM_PER_NORM
    error_y_cm = np.array(error_y_history, dtype=float) * CM_PER_NORM
    distance_cm = np.sqrt(error_x_cm**2 + error_y_cm**2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'Ball Position Error with PID Control '
    f'(P = {K_P_ROLL}, I = {K_I_ROLL}, D = {K_D_ROLL})', fontsize=15, fontweight='bold')

    # ── Plot 1: Linear distance from center ───────────────────────────────
    ax1.plot(ball_time_history, distance_cm, color='mediumpurple', linewidth=2, label='Distance from Center')
    ax1.axhline(0.0, color='black',  linestyle='--', linewidth=1.5, label='Target (0 cm)')
    ax1.axhline( 1.5, color='green', linestyle='--', linewidth=1.5, label='Convergence Bound (±1.5 cm)')
    ax1.axhline(-1.5, color='green', linestyle='--', linewidth=1.5)
    ax1.set_title('Linear Distance from Center', fontsize=13)
    ax1.set_ylabel('Distance (cm)', fontsize=12)
    ax1.legend(loc='upper right', frameon=True, shadow=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ── Plot 2: X and Y error ─────────────────────────────────────────────
    ax2.plot(ball_time_history, error_x_cm, color='teal',      linewidth=2, label='X Error')
    ax2.plot(ball_time_history, error_y_cm, color='darkorange', linewidth=2, label='Y Error')
    ax2.axhline(0.0, color='black',  linestyle='--', linewidth=1.5, label='Target (0 cm)')
    ax2.axhline( 1.5, color='green', linestyle='--', linewidth=1.5, label='Convergence Bound (±1.5 cm)')
    ax2.axhline(-1.5, color='green', linestyle='--', linewidth=1.5)
    ax2.set_title('X and Y Distance from Center', fontsize=13)
    ax2.set_xlabel('Time Elapsed (Seconds)', fontsize=12)
    ax2.set_ylabel('Distance (cm)', fontsize=12)
    ax2.legend(loc='upper right', frameon=True, shadow=True)
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    print("\nDisplaying ball positioning performance graph...")
    plt.show()

# ─── MAIN BALANCING EXECUTION LOOP ────────────────────────────────────────
def main():
    # Collect ball type from the user up front (used in the CSV filename)
    prompt_for_ball_type()

    # Setup standard communication connection
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)
    if not portHandler.openPort() or not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError("Failed to build connection to controller hardware.")

    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    
    # Enable torque on all servos
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 1)

    # State variables for Derivative calculation tracking
    prev_error_x = 0.0
    prev_error_y = 0.0
    prev_time = time.time()

    integral_error_x = 0.0
    integral_error_y = 0.0

    last_valid_x = 0.0
    last_valid_y = 0.0
    lost_frame_count = 0
    MAX_LOST_FRAMES_HOLD = 10 # How many frames to trust old data before giving up
            
    # Loop rate control settings (~100Hz loop speed)
    loop_hz = 100.0
    loop_period = 1.0 / loop_hz

    print(f"\n--> Commencing Active Closed-Loop Balancing at Z={NEUTRAL_Z}mm...")
    print("Press Ctrl+C to terminate cleanly.")

    try:
        # Home Postion
        q_initial = [188, 162, 172]
        goto_home_via_position_mode(q_initial, packetHandler=packetHandler, portHandler=portHandler)
        time.sleep(0.5)
        
        session_start_time = time.time()
        
        while True:
            t_start = time.time()
            dt = t_start - prev_time
            if dt <= 0.0:
                dt = 0.001 # Prevent divide-by-zero errors

            ball_x, ball_y = get_ball_position()
            ball_x = -ball_x  # Invert X if camera is mounted flipped
            # ball_y = -ball_y  # Invert Y if camera is mounted flipped
            
            if ball_x == 0.0 and ball_y == 0.0:
                lost_frame_count += 1
                
                if lost_frame_count <= MAX_LOST_FRAMES_HOLD:
                    ball_x = last_valid_x
                    ball_y = last_valid_y
                    
                    log_this_frame = False 
                else:
                    ball_x = 0.0
                    ball_y = 0.0
                    log_this_frame = True
            else:
                lost_frame_count = 0
                
                last_valid_x = ball_x
                last_valid_y = ball_y
                log_this_frame = True

            # Center target coordinates are (0,0), so error is directly current position
            error_x = ball_x 
            error_y = ball_y

            d_error_x = (error_x - prev_error_x) / dt
            d_error_y = (error_y - prev_error_y) / dt

            integral_error_x += error_x * dt
            integral_error_y += error_y * dt

            # Clamp Integral term to prevent windup
            INT_LIMIT = 3.0
            integral_error_x = np.clip(integral_error_x, -INT_LIMIT / K_I_ROLL, INT_LIMIT / K_I_ROLL)
            integral_error_y = np.clip(integral_error_y, -INT_LIMIT / K_I_PITCH, INT_LIMIT / K_I_PITCH)
            
            if log_this_frame:
                ball_time_history.append(t_start - session_start_time)
                error_x_history.append(error_x)
                error_y_history.append(error_y)

            # PID
            # Note: Tweak signs (+/-) depending on camera mounting orientation
            target_roll  = (K_P_ROLL * error_x) + (K_D_ROLL * d_error_x) + (K_I_ROLL * integral_error_x)
            target_pitch = (K_P_PITCH * error_y) + (K_D_PITCH * d_error_y) + (K_I_PITCH * integral_error_y)

            # Clamp output ranges to protect the platform from over-tilting
            target_roll  = np.clip(target_roll, -MAX_TILT_DEG, MAX_TILT_DEG)
            target_pitch = np.clip(target_pitch, -MAX_TILT_DEG, MAX_TILT_DEG)
            
            print(f"Target Roll: {target_roll:.2f}°, Target Pitch: {target_pitch:.2f}°")

            # PROCESS THROUGH KINEMATICS & SHIP POSITION DATA TO HARDWARE
            motor_angles, valid = inverse_kinematics(target_roll, target_pitch, NEUTRAL_Z)

            if valid:
                groupSyncWrite.clearParam()
                for i in range(3):
                    goal_ticks = deg_to_pos_ticks(motor_angles[i])
                    param_goal_position = [
                        DXL_LOBYTE(DXL_LOWORD(goal_ticks)), DXL_HIBYTE(DXL_LOWORD(goal_ticks)),
                        DXL_LOBYTE(DXL_HIWORD(goal_ticks)), DXL_HIBYTE(DXL_HIWORD(goal_ticks))
                    ]
                    groupSyncWrite.addParam(DXL_IDS[i], param_goal_position)
                
                groupSyncWrite.txPacket()
            else:
                print("Warning: Calculated balancing command hit local kinematic limits!")

            # Log data history variables for next frame differentiation pass
            prev_error_x = error_x
            prev_error_y = error_y
            prev_time = t_start

            # Enforce continuous strict loop execution cycle rhythm
            elapsed = time.time() - t_start
            if elapsed < loop_period:
                time.sleep(loop_period - elapsed)

    except KeyboardInterrupt:
        print("\nBalancing interrupted by user.")

    finally:
        print("Disabling motor holding torques safely...")
        groupSyncWrite.clearParam()
        for dxl_id in DXL_IDS:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
        portHandler.closePort()
        print("System shutdown clear.")
        
        save_data_to_csv()
        plot_ball_error_performance()

if __name__ == "__main__":
    main()