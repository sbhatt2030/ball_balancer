import cv2
import numpy as np, os, time
import time
import serial
import threading
import queue
import matplotlib.pyplot as plt
from B_P_Traj_lib import Trajectory
from ball_detection import detect_ball_x
from scipy.optimize import fsolve  # Not using IK for now

arduino = serial.Serial('COM3', 9600)
time.sleep(2)

debug = True            
debug_plot_plan = True      # To get PNG of the planned trajectory arrays once

# Logging for plot
time_log = []
xb_log = []
xd_log = []

# Servo limits
MIN_ANGLE = 0
MAX_ANGLE = 30
NEUTRAL_SERVO = 15

SCALE_M_PER_UNIT = 0.1423   # m per normalized unit (From camera calibration test)

# Mechanism geometry (m)
l = 0.2
l11 = 0.06
l12 = 0.08
w = 0.16
h = 0.12

# PID gains
Kp = 11.0
Ki = 3.0
Kd = 9.9

integral_error_pos = 0.0
previous_x_for_pid = None
velocity_filt = 0.0
VEL_ALPHA = 0.20    # Adjust based on servo jitter
INT_CLAMP = 0.5
g = 9.81

# Trajectory planner buffers
N_PLAN = 31               # Must be >= 31 because trajectory planner writes index 30 in some branches
pltt = np.zeros(N_PLAN)   # Time grid
plx  = np.zeros(N_PLAN)   # Scratch/segment pos (not used in PID)
posx = np.zeros(N_PLAN)   # Planned abs/rel pos
velx = np.zeros(N_PLAN)   # Planned vel
accx = np.zeros(N_PLAN)   # Planned acc

# Trajectory Limits
A_MAX = g * np.sin(np.radians(15.0))  # Max linear acc of ball at ±15° tilt: a = g·sinθ
A_MIN = -A_MAX
V_MAX = 0.5  # [m/s] chosen desired velocity cap
V_MIN = -V_MAX

# If pltt isn’t strictly increasing, synthesize a simple time base:
T_SYN_FALLBACK = 0.70  # seconds

data_queue = queue.Queue(maxsize=1)

# Reference signals shared between loops
ref_lock = threading.Lock()
xd_ref = 0.0
vdx_ref = 0.0
adx_ref = 0.0

plan_ready = False
plan_start_t = None
plan_x0 = 0.0  # absolute start position used to build the plan (for offsetting relative pos outputs)

# Optional replan behavior
replan_on_large_error = False
replan_error_thresh = 0.06  # meters

# Interpolation
def _interp(t, t0, t1, y0, y1):
    if t1 <= t0:  # degenerate guard
        return float(y0)
    a = (t - t0) / (t1 - t0)
    return float((1.0 - a) * y0 + a * y1)

# Substitue for Inverse Kinematics
def beam_angle_to_servo_angle(theta, neutral_servo):
    servo_deg = neutral_servo + np.degrees(theta)
    return servo_deg

# Arduino Communication
def send_servo_angle(servo_deg):
    servo_deg = int(np.clip(servo_deg, MIN_ANGLE, MAX_ANGLE))
    try:
        arduino.write(bytes([servo_deg]))
    except Exception as e:
        print(f"[Serial Error] {e}")

def reset_pid_state():
    global integral_error_pos, previous_x_for_pid, velocity_filt
    integral_error_pos = 0.0
    previous_x_for_pid = None
    velocity_filt = 0.0

# PID Controller (position only)
def controller(xb, xd=0.0, vbx=None, vdx=0.0, dt=0.01):
    global integral_error_pos, previous_x_for_pid, velocity_filt

    # Fallback vel estimate for controller
    if vbx is None:
        if previous_x_for_pid is None:
            vbx_raw = 0.0
        else:
            vbx_raw = (xb - previous_x_for_pid) / dt
        previous_x_for_pid = xb
    else:
        vbx_raw = vbx

    # Low-pass filter for vel
    velocity_filt = (1.0 - VEL_ALPHA) * velocity_filt + VEL_ALPHA * vbx_raw

    # PID on position; Kd term compares ref vel vs measured vel;  Restrict integral term
    error = xd - xb
    integral_error_pos = np.clip(integral_error_pos + error * dt, -INT_CLAMP, INT_CLAMP)
    a_cmd = (Kp * error) + (Ki * integral_error_pos) + (Kd * (vdx - velocity_filt))

    # Dynamics inversion: a = (5/7) g sin(theta) → theta
    s = np.clip((7.0/5.0) * (a_cmd / g), -0.99, 0.99)
    theta_cmd = float(np.arcsin(s))  # radians

    # Safety clamp on beam tilt (+- 15 degrees)
    max_tilt_rad = np.radians(15.0)
    theta = float(np.clip(theta_cmd, -max_tilt_rad, max_tilt_rad))
    return theta

# Trajectory planner
def build_plan(xb0, vb0, xd_goal=0.0, vdx_goal=0.0):
    """
    Fill pltt, posx, velx, accx using Trajectory().
    Store plan_x0 to offset relative outputs to absolute camera frame.
    """
    global plan_x0
    plan_x0 = float(xb0)

    # Clear arrays
    pltt.fill(0.0); plx.fill(0.0); posx.fill(0.0); velx.fill(0.0); accx.fill(0.0)

    _ = Trajectory(xb0, xd_goal, vb0, vdx_goal,
                   A_MAX, A_MIN, V_MAX, V_MIN,
                   plx, pltt, posx, velx, accx)

    # Guard to ensure strictly increasing time axis for interpolation
    if not np.all(np.diff(pltt) > 0):
        n = len(pltt)
        pltt[:] = np.linspace(0.0, T_SYN_FALLBACK, n)

    if debug:
        print(f"[PLAN ARRAYS] posx: min={np.min(posx):+.4f} m  max={np.max(posx):+.4f} m  "
            f"start={posx[0]:+.4f} m  end={posx[-1]:+.4f} m  Δ={posx[-1]-posx[0]:+.4f} m")
        print(f"[PLAN ARRAYS] velx: min={np.min(velx):+.4f} m/s  max={np.max(velx):+.4f} m/s")
        print(f"[PLAN ARRAYS] accx: min={np.min(accx):+.4f} m/s² max={np.max(accx):+.4f} m/s²")

    # Write one-time debug PNG to inspect planned arrays
    if debug_plot_plan:
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            import os

            script_dir = os.path.dirname(os.path.abspath(__file__))
            png_path   = os.path.join(script_dir, "Planned_trajectory_debug.png")

            fig = Figure(figsize=(6, 4), dpi=120)
            _ = FigureCanvas(fig)  # non-interactive Agg canvas

            # Axes: left for position (mm), right for velocity/accel (SI)
            ax_pos = fig.add_subplot(111)
            ax_va  = ax_pos.twinx()

            # Position relative to start (helps reveal curvature), in mm
            delta_pos_mm = 1000.0 * (posx - posx[0])
            print(f"[PLAN DEBUG] dp_end={delta_pos_mm[-1]:.1f} mm  range=[{np.min(delta_pos_mm):.1f}, {np.max(delta_pos_mm):.1f}] mm")

            # Plots
            p1, = ax_pos.plot(pltt, delta_pos_mm, color="tab:blue", label="Planned Position Δx (mm)")
            p2, = ax_va.plot(pltt, velx, color="tab:orange", label="Planned Velocity (m/s)",  linewidth=2)
            p3, = ax_va.plot(pltt, accx, color="tab:green", label="Planned Accel (m/s²)",   linewidth=2)

            # Labels / grid / legend
            ax_pos.set_xlabel("Time (s)")
            ax_pos.set_ylabel("Position (mm)")
            ax_va.set_ylabel("Velocity (m/s), Accel (m/s²)")
            ax_pos.grid(True, which="both", alpha=0.3)
            fig.suptitle("Planned trajectory arrays")

            handles = [p1, p2, p3]
            labels  = [h.get_label() for h in handles]
            ax_pos.legend(handles, labels, loc="upper left")

            # Save PNG (thread-safe, non-GUI)
            fig.savefig(png_path, bbox_inches="tight")
            print(f"[PLAN] saved {png_path} (cwd={os.getcwd()})")

            if debug:
                print("[PLAN] saved planned_trajectory.png")
        except Exception as e:
            print(f"[PLAN PLOT ERROR] {e}")

# Trajectory loop (Plans time-based references)
def trajectory_loop():
    global xd_ref, vdx_ref, adx_ref

    while True:
        if not plan_ready:
            time.sleep(0.005)
            continue

        # Elapsed time on the plan
        t = time.monotonic() - plan_start_t
        if t <= 0.0:
            t = 0.0
        if t >= pltt[-1]:
            t = pltt[-1]

        # Bracket and interpolate
        n = len(pltt)
        i  = int(np.searchsorted(pltt, t, side="left"))
        i0 = max(0, min(n - 2, i - 1))
        i1 = i0 + 1
        t0, t1 = pltt[i0], pltt[i1]

        # Position may be relative to 0; make it absolute in camera frame if needed
        pos_sample = _interp(t, t0, t1, posx[i0], posx[i1])
        if abs(posx[0]) < 1e-12:
            xd  = plan_x0 + pos_sample
        else:
            xd  = pos_sample

        vdx = _interp(t, t0, t1, velx[i0], velx[i1])
        adx = _interp(t, t0, t1, accx[i0], accx[i1])

        with ref_lock:
            xd_ref  = float(xd)
            vdx_ref = float(vdx)
            adx_ref = float(adx)

        time.sleep(0.005)

# Control loop
def control_loop():
    global plan_ready, plan_start_t, previous_x_for_pid

    prev_frame_time = None
    xb, vbx = 0.0, 0.0
    first_sample_seen = False

    # Start at neutral
    send_servo_angle(NEUTRAL_SERVO)

    while True:
        try:
            x, frame_time = data_queue.get(timeout=0.1)

            # dt from camera timestamps
            if prev_frame_time is None:
                dt_obs = 0.01
            else:
                dt_obs = max(1e-3, frame_time - prev_frame_time)
            prev_frame_time = frame_time

            # Measured velocity
            vbx = (x - xb) / dt_obs
            xb = x

            # First valid sample: zero first velocity and build the plan once
            if not first_sample_seen:
                vbx = 0.0
                previous_x_for_pid = xb   # For PID's internal vel estimate
                build_plan(xb0=xb, vb0=vbx, xd_goal=0.0, vdx_goal=0.0)
                plan_start_t = time.monotonic()
                plan_ready = True
                first_sample_seen = True

            # Read current references from the trajectory loop
            with ref_lock:
                xd  = xd_ref
                vdx = vdx_ref

            # Optional: replan if error is very large (disabled by default)
            if replan_on_large_error:
                e = xd - xb
                if abs(e) > replan_error_thresh:
                    build_plan(xb0=xb, vb0=vbx, xd_goal=0.0, vdx_goal=0.0)
                    with ref_lock:
                        plan_start_t = time.monotonic()
                        plan_ready = True
                    reset_pid_state()

            theta = controller(xb, xd, vbx, vdx, dt_obs)
            servo_deg_cmd = beam_angle_to_servo_angle(theta, NEUTRAL_SERVO)

            if debug:
                print(f"[CTRL] xb={xb:.3f}  xd={xd:.3f}  vbx={vbx:.3f}  vdx={vdx:.3f}  "
                      f"θ={np.degrees(theta):.1f}°  servo={servo_deg_cmd:.1f}°")

            if not time_log:
                time_log.append(0.0)
            else:
                time_log.append(time_log[-1] + dt_obs)
            xb_log.append(xb)
            xd_log.append(xd)

            send_servo_angle(servo_deg_cmd)

        except queue.Empty:
            # Keep last command if no fresh data; avoids bounce
            pass
        except Exception as e:
            print(f"[CONTROL LOOP ERROR] {e}")

# Camera loop
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Start both loops: PID and trajectory planner
    threading.Thread(target=control_loop,    daemon=True).start()
    threading.Thread(target=trajectory_loop, daemon=True).start()

    global plan_ready, plan_start_t
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        found, x_raw, _ = detect_ball_x(frame)
        x = x_raw * SCALE_M_PER_UNIT        # Scaled to meters

        if found:
            try:
                if data_queue.full():
                    data_queue.get_nowait()
                data_queue.put_nowait((x, time.monotonic()))
            except queue.Empty:
                pass

        cv2.imshow("Ball Tracker", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key to exit
            send_servo_angle(NEUTRAL_SERVO)
            try:
                reset_pid_state()
            except Exception:
                pass
            break
        elif k == ord('r'):
            # Hit 'r' key to start manual replan from current measured state
            try:
                x_now, _ = data_queue.get_nowait()
                xb_now = x_now
            except Exception:
                xb_now = 0.0
            build_plan(xb0=xb_now, vb0=0.0, xd_goal=0.0, vdx_goal=0.0)
            with ref_lock:
                plan_start_t = time.monotonic()
                plan_ready = True
            reset_pid_state()
            if debug:
                print("[UI] Replanned trajectory from current state")

    cap.release()
    cv2.destroyAllWindows()

    # Plot
    plt.figure()
    plt.plot(time_log, xb_log, label="Current Position (xb)")
    plt.plot(time_log, xd_log, label="Desired Position (xd)", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.2, 0.2)
    plt.title("Ball position vs planned reference")
    plt.show()

if __name__ == "__main__":
    main()
