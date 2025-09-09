# Minimal camera loop to calibrate the ball position scale coming from detect_ball_x().
# Shows current x [meters], running min/max, and how that compares to expected half-length (0.01m).
# - Press 's' to print a summary; 'r' to reset stats; 'q' or ESC to quit.

import cv2
import numpy as np
import time
from datetime import datetime

try:
    from ball_detection import detect_ball_x  # expected to return (found: bool, x_meters: float)
except Exception as e:
    print("[ERROR] Could not import detect_ball_x from ball_detection.py:", e)
    raise

BEAM_HALF_M = 0.10  # meters, expected half-length (center to one end)

CAM_INDEX = 0
FRAME_W, FRAME_H = 320, 240

def overlay_text(img, text, y, scale=0.6, thickness=1):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("[ERROR] Could not open camera index", CAM_INDEX)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    x_min, x_max = np.inf, -np.inf
    samples = 0
    t0 = time.time()

    # CSV log file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"c:/Users/varun/Downloads{ts}.csv"
    with open(csv_path, "w") as f:
        f.write("timestamp_s,found,x_m\n")

    print("[INFO] Move the ball gently to BOTH ends of the beam, then back to center.")
    print("[INFO] Press 's' anytime for a summary, 'r' to reset, 'q' or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Camera frame not ok; retrying...")
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))

        SCALE_M_PER_UNIT = 0.1423  # calibrated from multiple tests

        found, x_raw, _ = detect_ball_x(frame)  # raw value in normalized units (~-1..+1)
        x_m = x_raw * SCALE_M_PER_UNIT          # convert to meters
        
        now = time.time() - t0
        if found and np.isfinite(x_m):
            samples += 1
            x_min = min(x_min, x_m)
            x_max = max(x_max, x_m)

            # Append to CSV
            with open(csv_path, "a") as f:
                f.write(f"{now:.3f},{int(found)},{x_m:.6f}\n")

        # Draw UI
        frame_vis = frame.copy()
        overlay_text(frame_vis, f"x = {x_m:.4f} m   (found={found})", 20)
        if samples > 0:
            rng = x_max - x_min
            # Compare endpoints to expected half-length
            left_err  = (abs(x_min) - BEAM_HALF_M) if x_min < 0 else (x_min - (-BEAM_HALF_M))
            right_err = (x_max - BEAM_HALF_M) if x_max > 0 else ((-BEAM_HALF_M) - x_max)
            overlay_text(frame_vis, f"min = {x_min:.4f} m   max = {x_max:.4f} m   range = {rng:.4f} m", 40)
            overlay_text(frame_vis, f"expected half = {BEAM_HALF_M:.3f} m   left_err ~ {left_err:+.4f}   right_err ~ {right_err:+.4f}", 60)
        else:
            overlay_text(frame_vis, f"Collecting samples...", 40)

        overlay_text(frame_vis, "Press 's' summary | 'r' reset | 'q'/ESC quit", FRAME_H - 10, 0.55)

        cv2.imshow("Position Scale Probe", frame_vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('r'):
            x_min, x_max = np.inf, -np.inf
            samples = 0
            print("[INFO] Stats reset.")
        elif key == ord('s'):
            print_summary(x_min, x_max, samples, BEAM_HALF_M)

    cap.release()
    cv2.destroyAllWindows()
    print_summary(x_min, x_max, samples, BEAM_HALF_M)
    print(f"[INFO] CSV log saved to: {csv_path}")
    print("[DONE]")

def print_summary(x_min, x_max, samples, half_m):
    if samples == 0 or not np.isfinite(x_min) or not np.isfinite(x_max):
        print("[SUMMARY] Not enough data yet. Move the ball to both ends and press 's' again.")
        return
    rng = x_max - x_min
    left_ok  = np.isclose(abs(x_min), half_m, rtol=0.05, atol=0.01)  # within ~5% or 1 cm
    right_ok = np.isclose(abs(x_max), half_m, rtol=0.05, atol=0.01)

    print("\n========== POSITION SCALE SUMMARY ==========")
    print(f"samples: {samples}")
    print(f"x_min: {x_min:.4f} m (expected ~ {-half_m:.3f})  -> {'OK' if left_ok else 'OFF'}")
    print(f"x_max: {x_max:.4f} m (expected ~ {+half_m:.3f})  -> {'OK' if right_ok else 'OFF'}")
    print(f"range:  {rng:.4f} m (expected ~ {2*half_m:.3f})")
    if not left_ok or not right_ok:
        print("Hint: If OFF by a lot, your pixel->meter mapping or camera FOV calibration is wrong.")
    print("===========================================\n")

if __name__ == "__main__":
    main()
