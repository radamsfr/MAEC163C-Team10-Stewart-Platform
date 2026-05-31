import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- PHYSICAL DIMENSIONS  ---
R_BASE  = 87.5      # Radius of motor shaft mounting circle
R_PLATE = 129.5       # Radius of top plate ball-joint circle
L_ARM   = 42.0       # Length of servo arm/horn
L_ROD   = 64.0      # Length of connecting rod

# --- DEFINE FIXED BASE & MOVING PLATE COUPLING POINTS ---
# motors/joints spaced symmetrically at 0°, 120°, and 240°
angles = np.radians([0, 120, 240])

# Base anchor coordinates (3x3 matrix: columns are legs 1, 2, 3)
B = np.array([R_BASE * np.cos(angles), 
              R_BASE * np.sin(angles), 
              np.zeros(3)])

# Local plate coordinates (before translation/rotation)
P_local = np.array([R_PLATE * np.cos(angles), 
                    R_PLATE * np.sin(angles), 
                    np.zeros(3)])

# Orientation angles for the 3 servo rotation planes
beta = angles

# --- PHYSICAL USER HARDWARE CALIBRATION BOUNDS ---
Q_MAX_HORIZONTAL = np.array([188.0, 162.0, 172.0]) 
Q_MIN_VERTICAL   = np.array([98.0, 72.0, 82.0])    

# --- SETUP INTERACTIVE PLOT ENVIRONMENT ---
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.3)  # Leave room for sliders at bottom

# Configure sliders
ax_roll  = plt.axes([0.15, 0.15, 0.65, 0.03])
ax_pitch = plt.axes([0.15, 0.10, 0.65, 0.03])
ax_z     = plt.axes([0.15, 0.05, 0.65, 0.03])

s_roll  = Slider(ax_roll, 'Roll (deg)', -15.0, 15.0, valinit=0.0)
s_pitch = Slider(ax_pitch, 'Pitch (deg)', -15.0, 15.0, valinit=0.0)
s_z     = Slider(ax_z, 'Heave Z (mm)', 80.0, 150.0, valinit=115.0)

# --- CORE INVERSE KINEMATICS AND RENDERING FUNCTION ---
def update(val):
    r_deg = s_roll.val
    p_deg = s_pitch.val
    z_val = s_z.val

    r = np.radians(r_deg)
    p = np.radians(p_deg)

    # --- FIXED 3D ROTATION MATRIX SEQUENCE (Roll Rx * Pitch Ry) ---
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(r), -np.sin(r)],
        [0, np.sin(r), np.cos(r)]
    ])
    
    Ry = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])
    
    # Combined Rotation Matrix
    R = Rx @ Ry

    # Translate and Rotate top plate coordinates into world frame
    T = np.array([[0], [0], [z_val]])
    P_global = T + R @ P_local

    # Preserve custom camera rotation perspective across frames
    azim, elev = ax.azim, ax.elev
    ax.cla()
    ax.view_init(elev=elev, azim=azim)

    # Re-establish rigid frame boundary limits
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    ax.set_zlim(0, 200)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')

    # --- Compute Points for a Smooth Base Circle ---
    circle_angles = np.linspace(0, 2 * np.pi, 100)
    circle_x = R_BASE * np.cos(circle_angles)
    circle_y = R_BASE * np.sin(circle_angles)
    circle_z = np.zeros_like(circle_x) # Keep it flat on the floor

    # Draw the smooth base circle (thin gray dashed line)
    ax.plot(circle_x, circle_y, circle_z, color='gray', linestyle=':', lw=1, label='Base Boundary')

    # Draw Fixed Base Outline
    base_closed_x = np.append(B[0], B[0, 0])
    base_closed_y = np.append(B[1], B[1, 0])
    base_closed_z = np.append(B[2], B[2, 0])
    ax.plot(base_closed_x, base_closed_y, base_closed_z, 'k--', lw=1.5, label='Base Frame')

    # Draw Moving Top Plate Outline
    plate_closed_x = np.append(P_global[0], P_global[0, 0])
    plate_closed_y = np.append(P_global[1], P_global[1, 0])
    plate_closed_z = np.append(P_global[2], P_global[2, 0])
    ax.plot(plate_closed_x, plate_closed_y, plate_closed_z, 'b-', lw=2, label='Top Plate')

    motor_angles_physical = np.zeros(3)
    is_pose_valid = True

    # Loop to solve kinematic intersections for each leg
    for i in range(3):
        V = P_global[:, i] - B[:, i]
        
        # Project V onto the 2D servo plane
        x_proj = V[0] * np.cos(beta[i]) + V[1] * np.sin(beta[i])
        z_proj = V[2]

        rho_sq = x_proj**2 + z_proj**2
        val_cos = (rho_sq + L_ARM**2 - L_ROD**2) / (2 * L_ARM * np.sqrt(rho_sq))

        if abs(val_cos) <= 1.0:
            alpha = np.arccos(val_cos)
            gamma = np.arctan2(z_proj, x_proj)
            
            # Geometric calculation (Outward pointing arms layout)
            theta_math = gamma - alpha
            theta_deg_relative = np.degrees(theta_math)

            # --- FIXED MOTOR DIRECTION SIGN ---
            # As the arm moves UPWARDS (theta_math increases/positive), 
            # we subtract it from Q_MAX to force the physical value down toward Q_MIN.
            motor_angles_physical[i] = Q_MAX_HORIZONTAL[i] - theta_deg_relative

            # Software workspace soft-stop check
            if (motor_angles_physical[i] < Q_MIN_VERTICAL[i]) or (motor_angles_physical[i] > Q_MAX_HORIZONTAL[i]):
                is_pose_valid = False
                line_color = 'magenta'
                line_style = ':'
            else:
                line_color = 'green'
                line_style = '-'

            # Compute actual absolute 3D spatial coordinate of the servo horn tip
            A_local_x = L_ARM * np.cos(theta_math)
            A_global = B[:, i] + np.array([A_local_x * np.cos(beta[i]), 
                                           A_local_x * np.sin(beta[i]), 
                                           L_ARM * np.sin(theta_math)])

            # Plot Actuator Servo Horn (Red line)
            ax.plot([B[0, i], A_global[0]], [B[1, i], A_global[1]], [B[2, i], A_global[2]], 'r-o', lw=3)
            # Plot Connecting Tie-Rod (Green if valid, Dotted Magenta if boundary error)
            ax.plot([A_global[0], P_global[0, i]], [A_global[1], P_global[1, i]], [A_global[2], P_global[2, i]], color=line_color, ls=line_style, lw=2)
        else:
            is_pose_valid = False
            ax.plot([B[0, i], P_global[0, i]], [B[1, i], P_global[1, i]], [B[2, i], P_global[2, i]], 'm:', lw=1.5)

    # Dynamic Console Output updates
    if is_pose_valid:
        print(f"Roll: {r_deg:5.1f}° | Pitch: {p_deg:5.1f}° | Z: {z_val:5.1f}mm -> Motor Poses: [{motor_angles_physical[0]:.1f}°, {motor_angles_physical[1]:.1f}°, {motor_angles_physical[2]:.1f}°]")
    else:
        print(f"*** LIMIT EXCEEDED *** Roll: {r_deg:5.1f}° | Pitch: {p_deg:5.1f}° | Z: {z_val:5.1f}mm -> Out of Workspace bounds!")

    fig.canvas.draw_idle()

# Attach callback functions to sliders
s_roll.on_changed(update)
s_pitch.on_changed(update)
s_z.on_changed(update)

# Initial frame computation trigger
update(None)

print("\nClick and drag within the 3D window to rotate view perspective freely.")
plt.show()