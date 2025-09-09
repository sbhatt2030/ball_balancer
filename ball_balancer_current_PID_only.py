import cv2
import numpy as np
import time
import serial
import threading
import queue
import matplotlib.pyplot as plt 
from B_P_Traj_lib import Trajectory
from ball_detection import detect_ball_x
from scipy.optimize import fsolve   # Not using IK for now

arduino = serial.Serial('COM3', 9600)
time.sleep(2)

debug = False   #Set to true if debugging

# For plotting
time_log = []
xb_log = []
xd_log = []

# Servo safe range (Modify as desired)
MIN_ANGLE = 0           
MAX_ANGLE = 30
NEUTRAL_SERVO = 15

SCALE_M_PER_UNIT = 0.1423  # from camera calibration test (m per normalized unit)

# Mechanism Geometry (in m)
l = 0.2         # Beam pivot-pivot
l11 = 0.06      # Link 1
l12 = 0.08      # Link 2
w = 0.16        # Horizontal offset
h = 0.12        # Vertical offset

# For PID Control 
Kp = 11.0      
Ki = 3.0       
Kd = 9.9

integral_error_pos = 0.0
previous_time = None
previous_error = None
previous_x_for_pid = None
velocity_filt = 0.0
VEL_ALPHA = 0.20       # low-pass for velocity in D-term
INT_CLAMP = 0.5        # anti-windup clamp (m·s)
g = 9.81               

# Interpolation
def _interp(t, t0, t1, y0, y1):
    if t1 <= t0:  # degenerate guard
        return float(y0)
    a = (t - t0) / (t1 - t0)
    return float((1.0 - a) * y0 + a * y1)

#def get_reference_trajectory(t, plx, pltt, posx, velx, accx, xb, vbx):
       
    a_max, a_min = 20.0, -20.0      # m/s^2     #a_max = g * sin(15°); 15 degrees is safe max beam titl
    vmax,  vmin  = 10.0, -10.0      # m/s       #v_max = sqrt(2 * a_max * 0.1)
    xd_goal, vdx_goal = 0.0, 0.0    # center & stop

    # Fill arrays for the next ~30 samples
    _xb, _xd, _vbx, _vdx, _ = Trajectory(
        xb, xd_goal, vbx, vdx_goal,
        a_max, a_min, vmax, vmin,
        plx, pltt, posx, velx, accx
    )

    # Interpolate at time t
    i = int(np.searchsorted(pltt, t, side="left"))
    i0 = max(0, min(len(pltt) - 2, i - 1))
    i1 = i0 + 1
    t0, t1 = pltt[i0], pltt[i1]

    xd  = _interp(t, t0, t1, plx[i0],  plx[i1])
    vdx = _interp(t, t0, t1, velx[i0], velx[i1])
    adx = _interp(t, t0, t1, accx[i0], accx[i1])

    return xd, vdx, adx

# PID Controller (Only position)
def controller(xb, xd=0.0, vbx=None, vdx=0.0, dt=0.01):
    global integral_error_pos, previous_time, previous_x_for_pid, velocity_filt

    # velocity estimate
    if vbx is None:
        if previous_x_for_pid is None:
            vbx_raw = 0.0
        else:
            vbx_raw = (xb - previous_x_for_pid) / dt
        previous_x_for_pid = xb
    else:
        vbx_raw = vbx

    # low-pass filter for velocity
    velocity_filt = (1.0 - VEL_ALPHA) * velocity_filt + VEL_ALPHA * vbx_raw

    # position PID → desired acceleration
    error = xd - xb
    #integral_error_pos = np.clip(integral_error_pos + error * dt, -INT_CLAMP, INT_CLAMP)
    integral_error_pos = (integral_error_pos + error * dt)
    a_cmd = (Kp * error) + (Ki * integral_error_pos) + (Kd * (vdx - velocity_filt))
    
    # dynamics inversion: a = (5/7) g sin(theta) → theta
    s = np.clip((7.0/5.0) * (a_cmd / g), -0.99, 0.99)
    theta_cmd = float(np.arcsin(s))  # radians

    # Safety clamp on beam tilt
    max_tilt_rad = np.radians(15.0)
    theta = float(np.clip(theta_cmd, -max_tilt_rad, max_tilt_rad))
    return theta    # In radians

def reset_pid_state():
    global integral_error_pos, previous_x_for_pid, velocity_filt
    integral_error_pos = 0.0
    previous_x_for_pid = None
    velocity_filt = 0.0

"""
# Inverse Kinematics
alpha = np.arctan2(h, w)   # Geometry constant
swh = np.sqrt(w**2 + h**2) # Geometry constant

def IK_equation(theta11, theta):
    
    #theta   = beam tilt angle (rad)
    #theta11 = motor joint angle (rad)
    
    LHS = (l12**2 - l11**2 - h**2 - w**2 - l**2) / swh
    RHS = (-2*l*np.cos(theta + alpha)
           + 2*l11*np.cos(theta11 + alpha)
           - (2*l*l11*np.cos(theta - theta11)) / swh)
    return LHS - RHS

def inverse_kin(theta, guess=0.0):
    
    # Solve for theta11 given theta
    sol = fsolve(lambda theta11: IK_equation(theta11, theta), guess)
    return float(sol[0])

def forward_kin(theta11, guess=0.0):
    
    # Solve for theta given theta11
    sol = fsolve(lambda theta: IK_equation(theta11, theta), guess)
    return float(sol[0])

"""
def beam_angle_to_servo_angle(theta, NEUTRAL_SERVO, debug=False):
    
    # Convert rad -> deg and offset around neutral (Assumes +ve beam tilt -> +ve servo increment)
    servo_deg = NEUTRAL_SERVO + np.degrees(theta)
    return servo_deg

# Arduino Communication
def send_servo_angle(servo_deg):
    servo_deg = int(np.clip(servo_deg, MIN_ANGLE, MAX_ANGLE))
    try:
        arduino.write(bytes([servo_deg]))
    except Exception as e:
        print(f"[Serial Error] {e}")

# Shared queue for detection results
data_queue = queue.Queue(maxsize=1)

# For camera
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Start control loop on a background thread
    threading.Thread(target=control_loop, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        found, x_raw, _ = detect_ball_x(frame)   
        x = x_raw * SCALE_M_PER_UNIT             # convert to meters

        if found:
            try:
                # drop stale if full, then put newest without blocking
                if data_queue.full():
                    data_queue.get_nowait()
                data_queue.put_nowait((x, time.monotonic()))
            except queue.Empty:
                pass

        # Optional preview
        cv2.imshow("Ball Tracker", frame)
        if cv2.waitKey(1) == 27:  # Press ESC key to exit
            send_servo_angle(NEUTRAL_SERVO)
            try:
                reset_pid_state()
            except Exception:
                pass
            break

        # Small delay helps UI smoothness; not part of control timing
        #time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

    # Show Plot at loop exit
    plt.figure()
    plt.plot(time_log, xb_log, label="Current Position (xb)")
    plt.plot(time_log, xd_log, label="Desired Position (xd)", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")  # or normalized units
    plt.title(f"Ball Balancer Response\nKp={Kp}, Ki={Ki}, Kd={Kd}")
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.15, 0.15)
    plt.show()

# Control loop thread
def control_loop():
    # State placeholders; Trajectory() expects xb/vbx even if not used in PID
    prev_frame_time = None
    xb, vbx = 0.0, 0.0
    plot_time = None

    # Start at neutral once
    send_servo_angle(NEUTRAL_SERVO)

    while True:
        try:
            x, frame_time = data_queue.get(timeout=0.1)

            # Compute dt based on camera timestamps (more stable than time.time())
            if prev_frame_time is None:
                dt_obs = 0.01
            else:
                dt_obs = max(1e-3, frame_time - prev_frame_time)
            prev_frame_time = frame_time

            # Measured velocity of the ball from observations
            vbx = (x - xb) / dt_obs
            xb = x

            # Run PID every cycle; desired position is 0.0 
            theta = controller(x, 0.0, vbx, 0.0, dt_obs)

            servo_deg_cmd = beam_angle_to_servo_angle(theta, NEUTRAL_SERVO)

            if debug:
                print(f"[CTRL] x={x:.3f}  xd={0.0}  θ_servo_raw={np.degrees(theta):.1f}°  servo_sent={servo_deg_cmd:.1f}°")

            # Start reference time on first frame for plotting
            if plot_time is None:
                plot_time = frame_time
            time_log.append(frame_time - plot_time)   # Time in seconds since start
            xb_log.append(xb)                  # Current position
            xd_log.append(0.0)                 # Desired position (currently 0.0 - replace with xd when using trajectory planner)

            send_servo_angle(servo_deg_cmd)
            
        except queue.Empty:
            # No fresh camera data; keep last command (safer than bouncing to neutral)
            pass
        except Exception as e:
            print(f"[CONTROL LOOP ERROR] {e}")

if __name__ == "__main__":
    main()
