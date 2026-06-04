import time
import numpy as np
from dynamixel_sdk import *
from ball_velocity_tracker import track_ping_pong_ball

# ─── HARDWARE & CALIBRATION SETUP ─────────────────────────────────────────
DXL_IDS          = [3, 2, 1]
BAUDRATE         = 57600
PROTOCOL_VERSION = 2.0
DEVICENAME       = "COM3"

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

# ─── PD CONTROLLER TUNING PARAMETERS ──────────────────────────────────────
# Note: Pitch controls Y-axis motion, Roll controls X-axis motion.
# Start with small K_P values and K_D at zero, then tune up incrementally.
K_P_ROLL  = 10.0    # Proportional gain for X -> Roll tilt
K_D_ROLL  = 6.0    # Derivative gain (damping) for X speed
K_I_ROLL  = 1.5    # Integral gain for Roll (optional, can help with steady-state error)

K_P_PITCH = 10.0    # Proportional gain for Y -> Pitch tilt
K_D_PITCH = 6.0    # Derivative gain (damping) for Y speed
K_I_PITCH = 1.5    # Integral gain for Pitch (optional, can help with steady-state error)


MAX_TILT_DEG = 8.0  # Hard soft-stop cap to keep angles gentle and stable

# ─── INTERACTION HELPER FUNCTIONS ──────────────────────────────────────────────────
def deg_to_pos_ticks(deg):
    return int(deg * (4096.0 / 360.0))

def pos_ticks_to_deg(ticks):
    return ticks * (360.0 / 4096.0)

def _signed_deg_err(current, target):
    err = (target - current + 180) % 360 - 180
    return err

def get_ball_position(camera_index=1):
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

    # --- STEP 1: CONVERT TO GRAYSCALE & BLUR ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # --- STEP 2: CREATE A BLACK MASK FOR THE BACKGROUND ---
    # We create a blank, pure black canvas matching our frame size
    bg_mask = np.zeros_like(gray)
    
    # Use HoughCircles to find your large, black circular platform layout automatically.
    # If the camera is fixed, you can also replace this with a static drawn circle!
    platform_circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=150, maxRadius=240
    )

    # Default platform coordinates if Hough Circles misses it on a frame
    plat_x, plat_y, plat_r = CENTER_X, CENTER_Y, 200 

    if platform_circles is not None:
        platform_circles = np.uint16(np.around(platform_circles))
        # Take the largest detected circle (your platform)
        plat_x, plat_y, plat_r = platform_circles[0][0]

    # Draw a pure white circle on our black canvas mask to serve as the "window"
    cv2.circle(bg_mask, (plat_x, plat_y), plat_r, 255, -1)

    # --- STEP 3: MASK THE GRAYSCALE IMAGE ---
    # Copy pixels ONLY where bg_mask is white. Everything else becomes pure black!
    # This completely deletes your white room background.
    masked_gray = cv2.bitwise_and(gray, gray, mask=bg_mask)

    # --- STEP 4: TRACK THE WHITE BALL INSIDE THE CLEAN ZONE ---
    # Run our regular binary thresholding on the newly cleaned image
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

    # --- STEP 5: DRAW UI ELEMENTS ON THE ORIGINAL DISPLAY FRAME ---
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
        
        # DEBUG TIP: Change 'frame' to 'thresh' below to see the isolated binary view!
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


# ─── MAIN BALANCING EXECUTION LOOP ────────────────────────────────────────
def main():
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
        while True:
            t_start = time.time()
            dt = t_start - prev_time
            if dt <= 0.0:
                dt = 0.001 # Prevent divide-by-zero errors

            # 1. READ current ball position data
            ball_x, ball_y = get_ball_position()
            
            ball_x = -ball_x  # Invert X if camera is mounted flipped
            # ball_y = -ball_y  # Invert Y if camera is mounted flipped

            # Center target coordinates are (0,0), so error is directly current position
            error_x = ball_x 
            error_y = ball_y

            d_error_x = (error_x - prev_error_x) / dt
            d_error_y = (error_y - prev_error_y) / dt

            integral_error_x += error_x * dt
            integral_error_y += error_y * dt

            # Clamp 
            INT_LIMIT = 3.0
            integral_error_x = np.clip(integral_error_x, -INT_LIMIT / K_I_ROLL, INT_LIMIT / K_I_ROLL)
            integral_error_y = np.clip(integral_error_y, -INT_LIMIT / K_I_PITCH, INT_LIMIT / K_I_PITCH)

            # 3. RUN THE PD ALGORITHM 
            # Note: Tweak signs (+/-) depending on camera mounting orientation!
            target_roll  = (K_P_ROLL * error_x) + (K_D_ROLL * d_error_x) + (K_I_ROLL * integral_error_x)
            target_pitch = (K_P_PITCH * error_y) + (K_D_PITCH * d_error_y) + (K_I_PITCH * integral_error_y)

            # Clamp output ranges to protect the platform from over-tilting
            target_roll  = np.clip(target_roll, -MAX_TILT_DEG, MAX_TILT_DEG)
            target_pitch = np.clip(target_pitch, -MAX_TILT_DEG, MAX_TILT_DEG)
            
            print(f"Target Roll: {target_roll:.2f}°, Target Pitch: {target_pitch:.2f}°")

            # 4. PROCESS THROUGH KINEMATICS & SHIP POSITION DATA TO HARDWARE
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

if __name__ == "__main__":
    main()