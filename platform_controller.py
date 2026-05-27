import os
import math
import time
from dynamixel_sdk import *

# Dynamixel IDs in order: (base joint, elbow joint).
# Check with Dynamixel Wizard. Convention used here is base = ID 1, elbow = ID 2.
DXL_IDS          = [3, 2, 1]
BAUDRATE         = 57600
PROTOCOL_VERSION = 2.0           # MX-28AR uses Protocol 2.0

# Serial port — adjust for your OS
# Linux:   '/dev/ttyUSB0'
# Mac:     '/dev/tty.usbserial-*'
# Windows: 'COM3'
DEVICENAME =  "COM3"

# ── Control Table Addresses (MX-28AR Protocol 2.0)
ADDR_OPERATING_MODE   = 11
ADDR_TORQUE_ENABLE    = 64
ADDR_GOAL_PWM         = 100
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION    = 116
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

# Byte lengths for sync read / write
LEN_GOAL_PWM         = 2
LEN_PRESENT_VELOCITY = 4
LEN_PRESENT_POSITION = 4

# Operating modes
PWM_CONTROL_MODE      = 16
POSITION_CONTROL_MODE = 3

PWM_FULL_SCALE = 885
SUPPLY_VOLTAGE = 12.0

# ── Conversions
DEG_PER_TICK = 360.0 / 4096                  # 0.0879°/tick
RAD_PER_TICK = (2.0 * math.pi) / 4096        # 0.001534 rad/tick
# Velocity register: 1 tick = 0.229 rev/min  =>  rad/s per tick = 0.229 * 2π / 60
VEL_RAD_PER_S_PER_TICK = 0.229 * 2.0 * math.pi / 60.0   # ≈ 0.02398

# ── Manipulator parameters (from CAD)
M1, M2   = 0.193537, 0.0156075   # link masses [kg]
LC1, LC2 = 0.0533903, 0.0281188  # link COM distances [m]
L1       = 0.0675                # link 1 length [m]

# ── Control loop rate
CONTROL_FREQ_HZ = 30.0           # 30 Hz is comfortable for sync-read/write of 2 motors
MAX_DURATION_S  = 2.0            # length of each trial

# ── DC-motor model parameters
KT_NM_PER_A = 1.78
R_OHM       = 8.57
KE_V_S_PER_RAD = 12.0 / 5.76     # ≈ 2.083  (≈ Kt for an ideal DC motor in SI)



# --- STEWART PLATFORM GEOMETRY CONSTANTS (in mm) ---
R_BASE = 0.0 #TODO          # Radius of the base anchor points circle
R_PLATFORM = 0.0 #TODO      # Radius of the top platform anchor points circle
LINK_ARM = 0.0 #TODO        # Length of the servo horn/arm
LINK_ROD = 0.0 #TODO        # Length of the connecting rod (pushrod)
INITIAL_Z = 0.0 #TODO       # Default resting height of the platform

# Angular positions of the 3 base servos (120 degrees apart)
BASE_ANGLES = [0, math.radians(120), math.radians(240)]


# --- INVERSE KINEMATICS FUNCTION ---
def calculate_servo_angles(roll, pitch, heave):
    """
    Calculates required servo angles (in radians) for a given Roll, Pitch, and Heave.
    """
    servo_angles = []
    
    # Rotation Matrix for Roll (alpha) and Pitch (beta)
    # Yaw is omitted or kept at 0 since this is a 3-DoF orientation platform
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
                    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
                    
    R = np.dot(R_y, R_x) # Combined rotation matrix

    for i in range(3):
        # 1. Coordinate of the base anchor point for this leg
        b_i = np.array([R_BASE * math.cos(BASE_ANGLES[i]), 
                        R_BASE * math.sin(BASE_ANGLES[i]), 
                        0.0])
                        
        # 2. Coordinate of the top platform anchor point relative to platform center
        p_home = np.array([R_PLATFORM * math.cos(BASE_ANGLES[i]), 
                           R_PLATFORM * math.sin(BASE_ANGLES[i]), 
                           0.0])
                           
        # 3. Transform top point by rotation matrix and add translation (Heave)
        p_i = np.dot(R, p_home) + np.array([0.0, 0.0, heave])
        
        # 4. Calculate effective leg vector from base joint to platform joint
        leg_vector = p_i - b_i
        L = np.linalg.norm(leg_vector)
        
        # 5. Triangle geometry to solve for the servo horn angle (Theta)
        # Using the law of cosines to find the angle of the servo arm
        x_i = leg_vector[0]
        y_i = leg_vector[1]
        z_i = leg_vector[2]
        
        # Closed-form geometric solution for a 3-DoF leg orientation
        # This simplifies the intersection of the servo arm circle and pushrod sphere
        # Adjust signs depending on whether your motor shafts point inward or outward
        expr = (L**2 + LINK_ARM**2 - LINK_ROD**2) / (2 * LINK_ARM * math.sqrt(x_i**2 + z_i**2))
        if abs(expr) > 1.0:
            raise ValueError(f"Target orientation physically unreachable for leg {i}!")
            
        theta = math.asin(expr) - math.atan2(x_i, z_i)
        servo_angles.append(theta)
        
    return servo_angles

def rad_to_dxl(rad):
    """Converts radians to Dynamixel 0-4095 position values centered at 2048 (0 rad)."""
    # 2048 is standard center position for X-series. Modify if using 0-1023 resolution motors.
    return int(2048 + (rad * (4095 / (2 * math.pi))))

# --- MAIN INITIALIZATION & RUN LOOP ---
def main():
    # Initialize PortHandler & PacketHandler
    portHandler = PortHandler(DEVICENAME)
    packetHandler = PacketHandler(PROTOCOL_VERSION)

    # Open port
    if not portHandler.openPort():
        print("Failed to open the port")
        return
    if not portHandler.setBaudRate(BAUDRATE):
        print("Failed to change the baudrate")
        return

    # Enable Torque for all servos
    for dxl_id in DXL_IDS:
        packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
    print("Servos initialized and torque enabled.")

    try:
        while True:
            # Simple demonstration: Oscillate the platform pitch and roll over time
            t = time.time()
            target_roll = math.radians(10 * math.sin(t * 2))  # Roll oscillates +/- 10 degrees
            target_pitch = math.radians(10 * math.cos(t * 2)) # Pitch oscillates +/- 10 degrees
            target_heave = INITIAL_Z                          # Hold constant resting height

            try:
                angles = calculate_servo_angles(target_roll, target_pitch, target_heave)
                
                # Write targets to each Dynamixel motor
                for idx, id in enumerate(DXL_IDS):
                    dxl_position = rad_to_dxl(angles[idx])
                    
                    # Write position value to servo
                    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(
                        portHandler, id, ADDR_GOAL_POSITION, dxl_position
                    )
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Comm error: {packetHandler.getTxRxResult(dxl_comm_result)}")
                    elif dxl_error != 0:
                        print(f"Servo error: {packetHandler.getRxPacketError(dxl_error)}")

            except ValueError as e:
                print(f"Kinematics Error: {e}")

            time.sleep(0.01) # 100Hz control loop cycle

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Disable Torque and close port safely on exit
        for dxl_id in DXL_IDS:
            packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        portHandler.closePort()
        print("Torque disabled and port closed cleanly.")

if __name__ == "__main__":
    import numpy as np # Required for vector math operations
    main()