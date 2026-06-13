"""
Compare ping-pong ball vs golf ball balancing performance.

Reads two CSV files (produced by main.py's save_data_to_csv) and plots
the linear distance-from-center vs time on the same set of axes.

Includes:
  - Manual time-sync offsets so you can align the moment the ball is released
    on each run (since the two runs won't start their loops at exactly the
    same point relative to "ball release").
  - Automatic clipping of both datasets to the same length (after sync) so
    they can be plotted together without index/length mismatch errors.
"""

import csv
from matplotlib import font_manager
import numpy as np
import matplotlib.pyplot as plt

# ─── FILE PATHS ─────────────────────────────────────────────────────────────
# Update these to point at the CSV files produced by main.py
PING_PONG_CSV = "/Users/juliachen/Desktop/plotz/P8_I2_D4_ping_pong.csv"
GOLF_CSV      = "/Users/juliachen/Desktop/plotz/variablestarting/bestfarP8_I2_D4_golf.csv"

# ─── MANUAL SYNC OFFSETS ─────────────────────────────────────────────────────
# Set these to the time (in seconds, from each file's own t=0) at which the
# ball was actually released in that run. Both datasets will be shifted so
# that this moment becomes t=0 for plotting/comparison purposes.
PING_PONG_RELEASE_TIME_S = 1.953452
GOLF_RELEASE_TIME_S      = 1.715174


def load_csv(filepath):
    """Loads time and distance columns from a CSV file produced by main.py."""
    times = []
    distances = []
    with open(filepath, mode="r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            times.append(float(row["time_s"]))
            distances.append(float(row["distance_cm"]))
    return np.array(times), np.array(distances)


def sync_and_clip(*args):
    """
    Accepts any number of (t, d, release_time) triples, e.g.:
 
        sync_and_clip(t1, d1, release1, t2, d2, release2, t3, d3, release3, ...)
 
    For each dataset:
      - Shifts the time axis so its release time becomes t=0
      - Drops any samples before the release moment
 
    Then clips ALL datasets to the same length (the shortest of the group)
    so they can be plotted together on shared axes.
 
    Returns a flat tuple: (t1, d1, t2, d2, t3, d3, ...) in the same order
    the datasets were passed in.
    """
    if len(args) % 3 != 0:
        raise ValueError("sync_and_clip expects arguments in groups of three: t, d, release_time")
 
    shifted = []
    for i in range(0, len(args), 3):
        t, d, release = args[i], args[i + 1], args[i + 2]
 
        t_shifted = t - release
        mask = t_shifted >= 0
 
        shifted.append((t_shifted[mask], d[mask]))
 
    # Clip all datasets to the same (shortest) length
    min_len = min(len(t) for t, _ in shifted)
 
    result = []
    for t, d in shifted:
        result.append(t[:min_len])
        result.append(d[:min_len])
 
    return tuple(result)

def compare_balls():
    t_pp, d_pp = load_csv(PING_PONG_CSV)
    t_golf, d_golf = load_csv(GOLF_CSV) 

    t_pp, d_pp, t_golf, d_golf = sync_and_clip(
        t_pp, d_pp, PING_PONG_RELEASE_TIME_S,
        t_golf, d_golf, GOLF_RELEASE_TIME_S,
    )

    d_pp = d_pp/1.75
    d_golf = d_golf/1.75

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_pp, d_pp, color="tab:blue", linewidth=2, label="Ping Pong Ball")
    ax.plot(t_golf, d_golf, color="tab:red", linewidth=2, label="Golf Ball")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")

    ax.set_title("Linear Distance from Center: Ping Pong vs Golf Ball", fontname="Times New Roman", fontsize=18, fontweight="bold")
    ax.set_xlabel("Time From Release (s)", fontname="Times New Roman",fontsize=14)
    ax.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)

    font = font_manager.FontProperties(family='Times New Roman',size=12) 
    ax.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()


def pdvspid():
    pd_path = "/Users/juliachen/Desktop/plotz/P8_I0_D4_golf.csv"
    pid_path = "/Users/juliachen/Desktop/plotz/variablestarting/bestfarP8_I2_D4_golf.csv"
    pd_release_time_s = 1.449323
    pid_release_time_s = 1.715174   

    t_pd, d_pd = load_csv(pd_path)
    t_pid, d_pid = load_csv(pid_path)

    t_pd, d_pd, t_pid, d_pid = sync_and_clip(
        t_pd, d_pd, pd_release_time_s,
        t_pid, d_pid, pid_release_time_s,
    )

    d_pd = d_pd/1.75
    d_pid = d_pid/1.75

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_pd, d_pd, color="tab:blue", linewidth=2, label="PD Controller")
    ax.plot(t_pid, d_pid, color="tab:red", linewidth=2, label="PID Controller")

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")

    ax.set_title("Linear Distance from Center: PD vs PID Controller", fontname="Times New Roman", fontsize=18, fontweight="bold")
    ax.set_xlabel("Time From Release (s)", fontname="Times New Roman",fontsize=14)
    ax.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)

    font = font_manager.FontProperties(family='Times New Roman',size=12) 
    ax.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()

def startingdist():
    far_path = "/Users/juliachen/Desktop/plotz/variablestarting/bestfarP8_I2_D4_golf.csv"
    med_path = "/Users/juliachen/Desktop/plotz/variablestarting/midwayP8_I2_D4_golf.csv"
    close_path = "/Users/juliachen/Desktop/plotz/variablestarting/closeP8_I2_D4_golf.csv"
 
    far_release_time_s = 1.715174
    med_release_time_s = 1.551335
    close_release_time_s = 1.451308
 
    t_far, d_far = load_csv(far_path)
    t_med, d_med = load_csv(med_path)
    t_close, d_close = load_csv(close_path)
 
    t_far, d_far, t_med, d_med, t_close, d_close = sync_and_clip(
        t_far, d_far, far_release_time_s,
        t_med, d_med, med_release_time_s,
        t_close, d_close, close_release_time_s,
    )
 
    d_far = d_far / 1.75
    d_med = d_med / 1.75
    d_close = d_close / 1.75
 
    font = font_manager.FontProperties(family='Times New Roman', size=12)
 
    fig, (ax_far, ax_med, ax_close) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
 
    # ── Far starting position subplot ───────────────────────────────────────
    ax_far.plot(t_far, d_far, color="tab:blue", linewidth=2, label="Far Starting Position")
    ax_far.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax_far.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")
    ax_far.set_title("Far Starting Position", fontname="Times New Roman", fontsize=14, fontweight="bold")
    ax_far.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)
    ax_far.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax_far.grid(True, linestyle=":", alpha=0.6)
 
    # ── Medium starting position subplot ────────────────────────────────────
    ax_med.plot(t_med, d_med, color="tab:green", linewidth=2, label="Medium Starting Position")
    ax_med.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax_med.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")
    ax_med.set_title("Medium Starting Position", fontname="Times New Roman", fontsize=14, fontweight="bold")
    ax_med.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)
    ax_med.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax_med.grid(True, linestyle=":", alpha=0.6)
 
    # ── Close starting position subplot ─────────────────────────────────────
    ax_close.plot(t_close, d_close, color="tab:red", linewidth=2, label="Close Starting Position")
    ax_close.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax_close.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")
    ax_close.set_title("Close Starting Position", fontname="Times New Roman", fontsize=14, fontweight="bold")
    ax_close.set_xlabel("Time From Release (s)", fontname="Times New Roman", fontsize=14)
    ax_close.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)
    ax_close.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax_close.grid(True, linestyle=":", alpha=0.6)
 
    fig.suptitle("Linear Distance from Center: Varied Starting Positions",
                  fontname="Times New Roman", fontsize=18, fontweight="bold")
 
    plt.tight_layout()
    plt.show()

def disturbance():
    golf_path = "/Users/juliachen/Desktop/plotz/variablestarting/multipleP8_I2_D4_golf.csv"
    pp_path   = "/Users/juliachen/Desktop/plotz/variablestarting/multipleP8_I2_D4_ping_pong.csv"
 
    golf_release_time_s = 0.000001
    pp_release_time_s   = 0.976052
 
    t_golf, d_golf = load_csv(golf_path)
    t_pp, d_pp = load_csv(pp_path)
 
    t_golf, d_golf, t_pp, d_pp = sync_and_clip(
        t_golf, d_golf, golf_release_time_s,
        t_pp, d_pp, pp_release_time_s,
    )
 
    d_golf = d_golf / 1.75
    d_pp = d_pp / 1.75
 
    font = font_manager.FontProperties(family='Times New Roman', size=12)
 
    fig, (ax_golf, ax_pp) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
 
    # ── Golf ball subplot ──────────────────────────────────────────────────
    ax_golf.plot(t_golf, d_golf, color="tab:blue", linewidth=2, label="Golf Ball")
    ax_golf.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax_golf.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")
    ax_golf.set_title("Golf Ball", fontname="Times New Roman", fontsize=14, fontweight="bold")
    ax_golf.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)
    ax_golf.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax_golf.grid(True, linestyle=":", alpha=0.6)
 
    # ── Ping pong ball subplot ─────────────────────────────────────────────
    ax_pp.plot(t_pp, d_pp, color="tab:red", linewidth=2, label="Ping Pong Ball")
    ax_pp.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Target (0 cm)")
    ax_pp.axhline(1, color="tab:green", linestyle="--", linewidth=1.5, label="Convergence Bound (+1 cm)")
    ax_pp.set_title("Ping Pong Ball", fontname="Times New Roman", fontsize=14, fontweight="bold")
    ax_pp.set_xlabel("Time From Release (s)", fontname="Times New Roman", fontsize=14)
    ax_pp.set_ylabel("Distance (cm)", fontname="Times New Roman", fontsize=14)
    ax_pp.legend(loc="upper right", frameon=True, shadow=True, prop=font)
    ax_pp.grid(True, linestyle=":", alpha=0.6)
 
    fig.suptitle("Linear Distance from Center with Disturbance",
                  fontname="Times New Roman", fontsize=18, fontweight="bold")
 
    plt.tight_layout()
    plt.show()

def main():
    #compare_balls()
    #pdvspid()
    startingdist()
    #disturbance()

if __name__ == "__main__":
    main()