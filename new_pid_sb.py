import cv2
import numpy as np
import time
import serial
import threading
import queue
import matplotlib.pyplot as plt
import json
from collections import deque

class BallDetector:
    """Ball detector using config.json HSV bounds"""
    def __init__(self, config):
        self.lower_bound = np.array(config['ball_detection']['lower_hsv'], dtype=np.uint8)
        self.upper_bound = np.array(config['ball_detection']['upper_hsv'], dtype=np.uint8)
        
    def detect_ball(self, frame):
        """Detect ball and return (found, center, radius, x_normalized)"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None, 0.0
        
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        if radius < 5 or radius > 100:
            return False, None, None, 0.0
        
        # Convert to normalized position using config
        center_x = frame.shape[1] // 2
        normalized_x = (x - center_x) / center_x
        
        return True, (int(x), int(y)), radius, float(np.clip(normalized_x, -1.0, 1.0))

class ImprovedPIDController:
    """Improved PID controller with better structure and anti-windup"""
    def __init__(self, config):
        # PID gains - can be tuned
        self.Kp = 11.0
        self.Ki = 3.0
        self.Kd = 9.9
        
        # Configuration from calibration
        self.scale_m_per_unit = config['camera_calibration']['pixel_to_meter_ratio'] * config['camera']['frame_width'] / 2
        self.position_limits = (config['operating_range']['position_min_m'], 
                               config['operating_range']['position_max_m'])
        
        # Control parameters
        self.integral_clamp = 0.5      # Anti-windup limit (m·s)
        self.velocity_alpha = 0.20     # Low-pass filter for velocity
        self.max_tilt_deg = 15.0       # Maximum beam tilt
        
        # State variables
        self.reset()
        
        # Physics constants
        self.g = 9.81  # gravity
        
    def reset(self):
        """Reset controller state"""
        self.integral_error = 0.0
        self.previous_position = None
        self.velocity_filtered = 0.0
        
    def update(self, current_position, desired_position=0.0, desired_velocity=0.0, dt=0.01):
        """
        Update PID controller
        
        Args:
            current_position: Current ball position in meters
            desired_position: Desired ball position in meters (default: center)
            desired_velocity: Desired ball velocity in m/s (default: 0)
            dt: Time step in seconds
            
        Returns:
            servo_angle: Servo angle in degrees
        """
        # Calculate velocity with better handling
        if self.previous_position is None:
            velocity_raw = 0.0
        else:
            velocity_raw = (current_position - self.previous_position) / dt
        self.previous_position = current_position
        
        # Low-pass filter velocity to reduce noise
        self.velocity_filtered = ((1.0 - self.velocity_alpha) * self.velocity_filtered + 
                                 self.velocity_alpha * velocity_raw)
        
        # PID calculations
        error = desired_position - current_position
        
        # Proportional term
        P_term = self.Kp * error
        
        # Integral term with anti-windup
        self.integral_error += error * dt
        self.integral_error = np.clip(self.integral_error, -self.integral_clamp, self.integral_clamp)
        I_term = self.Ki * self.integral_error
        
        # Derivative term (comparing desired vs actual velocity)
        velocity_error = desired_velocity - self.velocity_filtered
        D_term = self.Kd * velocity_error
        
        # Total control output (desired acceleration)
        acceleration_cmd = P_term + I_term + D_term
        
        # Convert acceleration to beam tilt using ball dynamics
        # For a ball on an inclined beam: a = (5/7) * g * sin(θ)
        # Therefore: θ = arcsin((7/5) * a/g)
        sin_theta = np.clip((7.0/5.0) * (acceleration_cmd / self.g), -0.99, 0.99)
        beam_tilt_rad = np.arcsin(sin_theta)
        
        # Convert to degrees and apply safety limits
        beam_tilt_deg = np.degrees(beam_tilt_rad)
        beam_tilt_deg = np.clip(beam_tilt_deg, -self.max_tilt_deg, self.max_tilt_deg)
        
        return beam_tilt_deg
    
    def set_gains(self, kp=None, ki=None, kd=None):
        """Update PID gains"""
        if kp is not None: self.Kp = kp
        if ki is not None: self.Ki = ki
        if kd is not None: self.Kd = kd

class ServoController:
    """Servo controller with config-based settings"""
    def __init__(self, config):
        self.port = config['servo']['port']
        self.neutral_angle = config['servo']['neutral_angle']
        self.min_angle = config['servo']['min_angle']
        self.max_angle = config['servo']['max_angle']
        
        self.arduino = None
        
    def connect(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(self.port, 9600)
            time.sleep(2)  # Arduino reset delay
            print(f"[SERVO] Connected to {self.port}")
            return True
        except Exception as e:
            print(f"[SERVO ERROR] {e}")
            return False
    
    def send_angle(self, beam_tilt_deg):
        """Send beam tilt as servo angle"""
        if not self.arduino:
            return False
        
        # Convert beam tilt to servo angle (assuming linear relationship)
        servo_angle = self.neutral_angle + beam_tilt_deg
        servo_angle = int(np.clip(servo_angle, self.min_angle, self.max_angle))
        
        try:
            self.arduino.write(bytes([servo_angle]))
            return True
        except Exception as e:
            print(f"[SERVO ERROR] {e}")
            return False
    
    def move_to_neutral(self):
        """Move to neutral position"""
        return self.send_angle(0.0)  # 0 degrees tilt
    
    def disconnect(self):
        """Disconnect and return to neutral"""
        if self.arduino:
            self.move_to_neutral()
            time.sleep(0.5)
            self.arduino.close()

class BallBalancerController:
    """Main ball balancer controller"""
    def __init__(self, config_file="config.json"):
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.ball_detector = BallDetector(self.config)
        self.pid_controller = ImprovedPIDController(self.config)
        self.servo = ServoController(self.config)
        
        # Camera settings
        self.cam_index = self.config['camera']['index']
        self.frame_w = self.config['camera']['frame_width']
        self.frame_h = self.config['camera']['frame_height']
        
        # Logging
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        
        # Communication queue
        self.data_queue = queue.Queue(maxsize=1)
        
        # Control settings
        self.debug = True
        self.setpoint = 0.0  # Center position
        
    def camera_loop(self):
        """Camera capture loop"""
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame = cv2.resize(frame, (320, 240))  # Reduce for speed
            found, center, radius, x_normalized = self.ball_detector.detect_ball(frame)
            
            if found:
                # Convert to meters using calibration
                x_meters = x_normalized * self.pid_controller.scale_m_per_unit
                
                try:
                    if self.data_queue.full():
                        self.data_queue.get_nowait()
                    self.data_queue.put_nowait((x_meters, time.monotonic()))
                except queue.Empty:
                    pass
            
            # Draw detection
            if found:
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.putText(frame, f"x: {x_normalized:.3f}", 
                           (center[0]-30, center[1]-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            cv2.imshow("Ball Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset PID
                self.pid_controller.reset()
                print("[RESET] PID controller reset")
            elif key == ord('c'):  # Change setpoint
                try:
                    new_setpoint = float(input("Enter new setpoint (meters): "))
                    self.setpoint = np.clip(new_setpoint, 
                                          self.pid_controller.position_limits[0],
                                          self.pid_controller.position_limits[1])
                    print(f"[SETPOINT] Changed to {self.setpoint:.4f}m")
                except ValueError:
                    print("[ERROR] Invalid setpoint")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def control_loop(self):
        """Main control loop"""
        if not self.servo.connect():
            print("[ERROR] Could not connect to servo")
            return
        
        prev_time = None
        start_time = time.monotonic()
        
        # Move to neutral initially
        self.servo.move_to_neutral()
        time.sleep(1.0)
        
        print("[CONTROL] Starting control loop...")
        
        while True:
            try:
                position, timestamp = self.data_queue.get(timeout=0.1)
                
                # Calculate dt
                if prev_time is None:
                    dt = 0.01
                else:
                    dt = max(0.001, timestamp - prev_time)
                prev_time = timestamp
                
                # Update PID controller
                beam_tilt = self.pid_controller.update(position, self.setpoint, 0.0, dt)
                
                # Send to servo
                self.servo.send_angle(beam_tilt)
                
                # Logging
                current_time = timestamp - start_time
                self.time_log.append(current_time)
                self.position_log.append(position)
                self.setpoint_log.append(self.setpoint)
                self.control_log.append(beam_tilt)
                
                if self.debug:
                    print(f"[CTRL] pos={position:.4f}m, sp={self.setpoint:.4f}m, "
                          f"tilt={beam_tilt:.2f}°, dt={dt:.3f}s")
                
            except queue.Empty:
                # No fresh data, keep last command
                pass
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[CONTROL ERROR] {e}")
        
        self.servo.disconnect()
    
    def run(self):
        """Run the ball balancer"""
        print(f"[INFO] Starting Ball Balancer")
        print(f"[INFO] Operating range: {self.pid_controller.position_limits[0]:.4f}m to {self.pid_controller.position_limits[1]:.4f}m")
        print(f"[INFO] PID gains: Kp={self.pid_controller.Kp}, Ki={self.pid_controller.Ki}, Kd={self.pid_controller.Kd}")
        print("[CONTROLS]")
        print("  ESC: Exit")
        print("  'r': Reset PID")
        print("  'c': Change setpoint")
        
        # Start control loop in background thread
        control_thread = threading.Thread(target=self.control_loop, daemon=True)
        control_thread.start()
        
        # Run camera loop in main thread
        try:
            self.camera_loop()
        except KeyboardInterrupt:
            pass
        
        # Show results
        self.plot_results()
    
    def plot_results(self):
        """Plot control performance"""
        if not self.time_log:
            print("[PLOT] No data to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Position tracking
        ax1.plot(self.time_log, self.position_log, label="Actual Position", linewidth=2)
        ax1.plot(self.time_log, self.setpoint_log, label="Setpoint", linestyle="--", linewidth=2)
        ax1.set_ylabel("Position (m)")
        ax1.set_title(f"Ball Balancer Performance (Kp={self.pid_controller.Kp}, Ki={self.pid_controller.Ki}, Kd={self.pid_controller.Kd})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Control effort
        ax2.plot(self.time_log, self.control_log, label="Beam Tilt", color="orange", linewidth=2)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Beam Tilt (degrees)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function"""
    import sys
    
    config_file = "config.json"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    
    try:
        controller = BallBalancerController(config_file)
        controller.run()
    except FileNotFoundError:
        print(f"[ERROR] Config file '{config_file}' not found!")
        print("Please run auto_calibrate.py first to generate config.json")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()