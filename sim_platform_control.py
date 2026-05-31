import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dynamixel_sdk import *

# ─── HARDWARE CONFIGURATION ───────────────────────────────────────────────
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

# ─── STEWART PLATFORM PHYSICAL GEOMETRY (in mm) ───────────────────────────
R_BASE  = 87.5      # Radius of motor shaft mounting circle
R_PLATE = 129.5       # Radius of top plate ball-joint circle
L_ARM   = 42.0       # Length of servo arm/horn
L_ROD   = 64.0      # Length of connecting rod

BASE_ANGLES  = np.radians([0, 120, 240])
B       = np.array([R_BASE * np.cos(BASE_ANGLES), R_BASE * np.sin(BASE_ANGLES), np.zeros(3)])
P_LOCAL = np.array([R_PLATE * np.cos(BASE_ANGLES), R_PLATE * np.sin(BASE_ANGLES), np.zeros(3)])


# Hardware Calibration Limits (Max/Horizontal vs Min/Vertical)
Q_MAX_HORIZONTAL = np.array([188.0, 162.0, 172.0]) 
Q_MIN_VERTICAL   = np.array([98.0, 72.0, 82.0])    

# ─── CONVERSION HELPER FUNCTIONS ───────────────────────────────────────────────────
def deg_to_pos_ticks(deg):
    return int(deg * (4096.0 / 360.0))

def pos_ticks_to_deg(ticks):
    return ticks * (360.0 / 4096.0)

def _signed_deg_err(current, target):
    err = (target - current + 180) % 360 - 180
    return err

# ─── HOMING TEMPLATE METHOD ──────────────────────────────────────
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


# ─── CIRCLE DEMO ──────────────────────────────────────
def run_circle_demo(portHandler, packetHandler, amplitude_deg=6.0, z_height_mm=115.0, duration_s=4.0):
    """
    Executes a smooth, continuous 360-degree circular orbit (wobble demo) 
    by modulating Roll and Pitch concurrently, then returns to neutral.
    """
    print(f"\n--> Starting Circle Demo (Tilt: {amplitude_deg}°, Height: {z_height_mm}mm, Time: {duration_s}s)...")
    
    # Initialize the SyncWrite interface
    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    
    # Timing variables for loop rate stability (~50Hz refresh rate)
    loop_rate_hz = 50.0
    time_step = 1.0 / loop_rate_hz
    total_steps = int(duration_s * loop_rate_hz)
    
    for step in range(total_steps):
        t_start = time.time()
        
        # Calculate current angle phase around the circle (0 to 2*pi)
        phase = (step / total_steps) * 2.0 * np.pi
        
        # Calculate matching Roll and Pitch targets
        target_roll  = amplitude_deg * np.cos(phase)
        target_pitch = amplitude_deg * np.sin(phase)
        
        # --- Run Inverse Kinematics Engine ---
        r = np.radians(target_roll)
        p = np.radians(target_pitch)
        
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        R = Rx @ Ry
        
        P_global = np.array([[0], [0], [z_height_mm]]) + R @ P_LOCAL
        
        groupSyncWrite.clearParam()
        is_pose_valid = True
        motor_angles_physical = np.zeros(3)
        
        for i in range(3):
            V = P_global[:, i] - B[:, i]
            x_proj = V[0] * np.cos(BASE_ANGLES[i]) + V[1] * np.sin(BASE_ANGLES[i])
            z_proj = V[2]
            
            rho_sq = x_proj**2 + z_proj**2
            val_cos = (rho_sq + L_ARM**2 - L_ROD**2) / (2 * L_ARM * np.sqrt(rho_sq))
            
            if abs(val_cos) <= 1.0:
                theta_math = np.arctan2(z_proj, x_proj) - np.arccos(val_cos)
                motor_angles_physical[i] = Q_MAX_HORIZONTAL[i] - np.degrees(theta_math)
                
                # Check hardware limits
                if (motor_angles_physical[i] < Q_MIN_VERTICAL[i]) or (motor_angles_physical[i] > Q_MAX_HORIZONTAL[i]):
                    is_pose_valid = False
                    break
                
                # Pack position ticks
                goal_ticks = deg_to_pos_ticks(motor_angles_physical[i])
                param_goal_position = [
                    DXL_LOBYTE(DXL_LOWORD(goal_ticks)), DXL_HIBYTE(DXL_LOWORD(goal_ticks)),
                    DXL_LOBYTE(DXL_HIWORD(goal_ticks)), DXL_HIBYTE(DXL_HIWORD(goal_ticks))
                ]
                groupSyncWrite.addParam(DXL_IDS[i], param_goal_position)
            else:
                is_pose_valid = False
                break
                
        # Send packets if valid, skip if bounds are broken to protect hardware
        if is_pose_valid:
            groupSyncWrite.txPacket()
        else:
            print("Notice: Demo path clipped slightly to protect hardware safety limits.")
            
        # Maintain rigid execution loop speed
        elapsed = time.time() - t_start
        if elapsed < time_step:
            time.sleep(time_step - elapsed)
            
    print("Circle Demo Complete. Returning to stable center.\n")
    

# ─── SIMULATION CONTROL LOOP INTERACTION FUNCTION ──────────────────────
def live_hardware_sim_loop(portHandler, packetHandler):
    """
    Launches the combined Matplotlib UI, processes real-time inverse kinematics,
    and streams goal updates directly to the motors.
    """
    # Initialize the SyncWrite interface for ultra-low latency updates
    groupSyncWrite = GroupSyncWrite(portHandler, packetHandler, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)

    # Build Matplotlib interface framework
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(bottom=0.3)

    ax_roll  = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_pitch = plt.axes([0.15, 0.10, 0.65, 0.03])
    ax_z     = plt.axes([0.15, 0.05, 0.65, 0.03])

    s_roll  = Slider(ax_roll, 'Roll (deg)', -12.0, 12.0, valinit=0.0)
    s_pitch = Slider(ax_pitch, 'Pitch (deg)', -12.0, 12.0, valinit=0.0)
    s_z     = Slider(ax_z, 'Heave Z (mm)', 50.0, 95.0, valinit=70.0)

    # Internal dynamic rendering frame callback
    def update(val):
        r = np.radians(s_roll.val)
        p = np.radians(s_pitch.val)
        z_val = s_z.val

        # Setup Rotation Matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        R = Rx @ Ry

        # Translate and scale platform locations
        P_global = np.array([[0], [0], [z_val]]) + R @ P_LOCAL

        # Lock camera frame
        azim, elev = ax.azim, ax.elev
        ax.cla()
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(-150, 150); ax.set_ylim(-150, 150); ax.set_zlim(0, 200)
        ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')

        # Render background guidelines
        ax.plot(np.append(B[0], B[0,0]), np.append(B[1], B[1,0]), np.append(B[2], B[2,0]), 'k--', lw=1.5)
        ax.plot(np.append(P_global[0], P_global[0,0]), np.append(P_global[1], P_global[1,0]), np.append(P_global[2], P_global[2,0]), 'b-', lw=2)

        groupSyncWrite.clearParam()
        is_pose_valid = True
        motor_angles_physical = np.zeros(3)

        for i in range(3):
            V = P_global[:, i] - B[:, i]
            x_proj = V[0] * np.cos(BASE_ANGLES[i]) + V[1] * np.sin(BASE_ANGLES[i])
            z_proj = V[2]

            rho_sq = x_proj**2 + z_proj**2
            val_cos = (rho_sq + L_ARM**2 - L_ROD**2) / (2 * L_ARM * np.sqrt(rho_sq))

            if abs(val_cos) <= 1.0:
                theta_math = np.arctan2(z_proj, x_proj) - np.arccos(val_cos)
                motor_angles_physical[i] = Q_MAX_HORIZONTAL[i] - np.degrees(theta_math)

                # Workspace software boundary checks
                if (motor_angles_physical[i] < Q_MIN_VERTICAL[i]) or (motor_angles_physical[i] > Q_MAX_HORIZONTAL[i]):
                    is_pose_valid = False
                
                # Transform targets into raw byte buffers
                goal_ticks = deg_to_pos_ticks(motor_angles_physical[i])
                param_goal_position = [
                    DXL_LOBYTE(DXL_LOWORD(goal_ticks)), DXL_HIBYTE(DXL_LOWORD(goal_ticks)),
                    DXL_LOBYTE(DXL_HIWORD(goal_ticks)), DXL_HIBYTE(DXL_HIWORD(goal_ticks))
                ]
                groupSyncWrite.addParam(DXL_IDS[i], param_goal_position)

                # Draw physical linkages visually
                A_local_x = L_ARM * np.cos(theta_math)
                A_global = B[:, i] + np.array([A_local_x * np.cos(BASE_ANGLES[i]), A_local_x * np.sin(BASE_ANGLES[i]), L_ARM * np.sin(theta_math)])
                ax.plot([B[0, i], A_global[0]], [B[1, i], A_global[1]], [B[2, i], A_global[2]], 'r-o', lw=3)
                ax.plot([A_global[0], P_global[0, i]], [A_global[1], P_global[1, i]], [A_global[2], P_global[2, i]], color='g' if is_pose_valid else 'm', lw=2)
            else:
                is_pose_valid = False

        # If safe, commit positions directly across serial bus
        if is_pose_valid:
            groupSyncWrite.txPacket()
            print(f"Command Output -> R: {s_roll.val:5.1f}° | P: {s_pitch.val:5.1f}° | Servos: {[f'{q:.1f}°' for q in motor_angles_physical]}")
        else:
            print("!!! Movement Request Aborted: Exceeds physical kinematic limits !!!")

        fig.canvas.draw_idle()

    # Bind active listeners
    s_roll.on_changed(update)
    s_pitch.on_changed(update)
    s_z.on_changed(update)

    # Run base frame generation pass
    update(None)
    plt.show()


# ─── INTERACTIVE MAIN FUNCTION ────────────────────────────────────────────
def main():
    q_initial = [188, 162, 172]              # Horizontal baseline flat state
    q_vertical = [q - 90 for q in q_initial]  # Standard midpoint neutral state
    
    portHandler   = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    if not portHandler.openPort():
        raise RuntimeError(f"Failed to open connection to {DEVICENAME}.")
    if not portHandler.setBaudRate(BAUDRATE):
        raise RuntimeError("Failed to set specified baud rate configuration.")
    print(f"Port communication active on {DEVICENAME}")

    for dxl_id in DXL_IDS:
        _, comm_result, _ = packetHandler.ping(portHandler, dxl_id)
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(f"Ping failed for ID={dxl_id}.")

    try:
        # Home Postion
        goto_home_via_position_mode(q_initial, packetHandler=packetHandler, portHandler=portHandler)
        time.sleep(1.0)

        # Run Circle Demo on boot
        run_circle_demo(portHandler, packetHandler, amplitude_deg=2.5, z_height_mm=70.0, duration_s=4.0)
        time.sleep(0.5)
        
        goto_home_via_position_mode(q_initial, packetHandler=packetHandler, portHandler=portHandler)
        time.sleep(1.0)

        #Start interactive desktop simulation loop control
        print("\nLaunching Live Interaction Windows... Use sliders to manipulate hardware.")
        live_hardware_sim_loop(portHandler, packetHandler)

    finally:
        # Secure safety shutdown sweep
        print("\nSystem exiting safely. Stripping torque states...")
        for dxl_id in DXL_IDS:
            try:
                packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0)
            except Exception:
                pass
        portHandler.closePort()
        print("Communications safely dropped.")

if __name__ == "__main__":
    main()