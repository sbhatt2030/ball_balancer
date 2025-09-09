import cv2
import numpy as np

Beam_HalfLength = 10  # Dimensionless
SCALE_CORRECTION = 1  # Ignore, scaling done in main script camera loop

def detect_ball_x(frame):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV Image", hsv)

    # Define HSV range for orange
    lower_orange = np.array([7, 169, 134])
    upper_orange = np.array([22, 254, 200])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Clean up mask
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Draw center of frame
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2
    cv2.drawMarker(frame, (center_x, center_y), (255, 255, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    cv2.putText(frame, "Center", (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0.0, frame   # Ball out of range

    # Find largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

    # Filter based on radius
    if radius < 5 or radius > 100:
        return False, 0.0, frame   # Ball out of range

    # Draw ball detection
    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
    cv2.putText(frame, f"x: {int(x)}", (int(x) - 30, int(y) - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Return normalized x position
    normalized_x = (x - center_x) / center_x
    x = normalized_x * SCALE_CORRECTION
    return True, float(np.clip(x, -Beam_HalfLength, Beam_HalfLength)), frame

def main():
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Use 0 or 1 

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        found, ball_x, vis_frame = detect_ball_x(frame)

        # Show final output
        cv2.imshow("Ball Detection", vis_frame)

        if found:
            print(f"Ball X = {ball_x:.3f} m")  # so ball_x is actually used

        if cv2.waitKey(1) == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
