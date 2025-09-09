"""
Complete Ball Balancer Calibration System

This system performs:
1. Color calibration for ball detection
2. Camera geometric calibration  
3. Automatic range finding using servo control
4. Exports complete config.json with all parameters

Workflow:
1. Color calibration - click on ball to sample HSV values
2. Camera calibration - click on beam endpoints 
3. Ball diameter measurement
4. Automatic range finding - servo moves beam to find min/max positions
5. Export complete configuration
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
import math
import serial
from collections import deque
import threading

class BallDetector:
    """Enhanced ball detector with configurable HSV bounds"""
    def __init__(self):
        # Default HSV bounds (will be updated during calibration)
        self.lower_bound = np.array([5, 150, 150], dtype=np.uint8)
        self.upper_bound = np.array([20, 255, 255], dtype=np.uint8)
        
    def update_hsv_bounds(self, lower, upper):
        """Update HSV detection bounds"""
        # Ensure bounds are numpy arrays with correct dtype
        self.lower_bound = np.array(lower, dtype=np.uint8)
        self.upper_bound = np.array(upper, dtype=np.uint8)
        
    def detect_ball(self, frame):
        """
        Detect ball and return (found, center, radius, x_normalized)
        x_normalized is compatible with existing ball_detection.py interface
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask with current bounds
        mask = cv2.inRange(hsv, self.lower_bound, self.upper_bound)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, None, None, 0.0
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        # Filter based on radius
        if radius < 5 or radius > 100:
            return False, None, None, 0.0
        
        # Calculate normalized x position (for compatibility)
        center_x = frame.shape[1] // 2
        normalized_x = (x - center_x) / center_x
        
        return True, (int(x), int(y)), radius, float(np.clip(normalized_x, -1.0, 1.0))

class ServoController:
    """Arduino servo controller"""
    def __init__(self, port='COM3', baudrate=9600):
        self.port = port
        self.arduino = None
        self.neutral_angle = 15
        self.min_angle = 0
        self.max_angle = 30
        
    def connect(self):
        """Connect to Arduino"""
        try:
            self.arduino = serial.Serial(self.port, 9600)
            time.sleep(2)  # Arduino reset delay
            print(f"[SERVO] Connected to {self.port}")
            return True
        except Exception as e:
            print(f"[SERVO ERROR] Could not connect: {e}")
            return False
    
    def send_angle(self, angle):
        """Send angle command to Arduino"""
        if not self.arduino:
            return False
        
        angle = int(np.clip(angle, self.min_angle, self.max_angle))
        try:
            self.arduino.write(bytes([angle]))
            return True
        except Exception as e:
            print(f"[SERVO ERROR] Failed to send angle: {e}")
            return False
    
    def move_to_neutral(self):
        """Move servo to neutral position"""
        return self.send_angle(self.neutral_angle)
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.arduino:
            self.move_to_neutral()
            time.sleep(0.5)
            self.arduino.close()
            self.arduino = None

class ComprehensiveCalibrator:
    def __init__(self):
        # Physical parameters
        self.BEAM_LENGTH_M = 0.02
        self.BALL_DIAMETER_M = 0.00425
        
        # Camera settings
        self.CAM_INDEX = 0
        self.FRAME_W, self.FRAME_H = 640, 480
        
        # Components
        self.ball_detector = BallDetector()
        self.servo = ServoController()
        
        # Calibration state
        self.current_frame = None
        self.ui_mode = "color_calibration"
        
        # Color calibration
        self.hsv_samples = []
        self.sample_radius = 3
        
        # Camera calibration
        self.peg_points = []
        self.beam_center_px = None
        self.beam_vector_px = None
        self.beam_length_px = None
        self.pixel_to_meter_ratio = None
        
        # Ball measurement
        self.ball_measurements = []
        self.ball_pixel_to_meter = None
        
        # Range finding
        self.position_min = None
        self.position_max = None
        self.usable_range_m = None
        self.range_positions = []
        
        self.range_finding_active = False
        self.range_finding_thread = None
        
        # State management
        self.calibration_complete = False
        
        self.instructions = {
            "color_calibration": "Click on the ball to sample HSV colors. Press 'c' when done.",
            "camera_calibration": "Click on the TWO PEGS that define beam endpoints.",
            "ball_measurement": "Place ball on beam. Press SPACE to measure diameter (3-5 times).",
            "range_finding": "Press 'a' to start automatic range finding with servo control.",
            "complete": "Calibration complete! Press 's' to save config.json"
        }
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for different calibration phases"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.ui_mode == "color_calibration":
                self.sample_hsv_at_point(x, y)
            elif self.ui_mode == "camera_calibration" and len(self.peg_points) < 2:
                self.add_peg_point(x, y)
    
    def sample_hsv_at_point(self, x, y):
        """Sample HSV values around clicked point"""
        if self.current_frame is None:
            return
        
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Sample region around click
        x1 = max(0, x - self.sample_radius)
        x2 = min(w, x + self.sample_radius + 1)
        y1 = max(0, y - self.sample_radius)
        y2 = min(h, y + self.sample_radius + 1)
        
        region = hsv[y1:y2, x1:x2]
        region_flat = region.reshape(-1, 3)
        
        for hsv_pixel in region_flat:
            self.hsv_samples.append(hsv_pixel.copy())
        
        print(f"[COLOR] Sampled {len(region_flat)} pixels. Total: {len(self.hsv_samples)}")
        self.update_hsv_bounds()
    
    def update_hsv_bounds(self):
        """Update HSV bounds from samples"""
        if not self.hsv_samples:
            return
        
        samples_array = np.array(self.hsv_samples)
        
        h_values = samples_array[:, 0]
        s_values = samples_array[:, 1]
        v_values = samples_array[:, 2]
        
        # Calculate bounds with margin
        h_margin = max(5, (np.max(h_values) - np.min(h_values)) * 0.1)
        s_margin = max(10, (np.max(s_values) - np.min(s_values)) * 0.15)
        v_margin = max(10, (np.max(v_values) - np.min(v_values)) * 0.15)
        
        lower_bound = [
            max(0, np.min(h_values) - h_margin),
            max(0, np.min(s_values) - s_margin),
            max(0, np.min(v_values) - v_margin)
        ]
        
        upper_bound = [
            min(179, np.max(h_values) + h_margin),
            min(255, np.max(s_values) + s_margin),
            min(255, np.max(v_values) + v_margin)
        ]
        
        self.ball_detector.update_hsv_bounds(lower_bound, upper_bound)
        print(f"[COLOR] Updated bounds: L{lower_bound} U{upper_bound}")
    
    def add_peg_point(self, x, y):
        """Add peg point for camera calibration"""
        self.peg_points.append((x, y))
        print(f"[CAMERA] Peg {len(self.peg_points)} at ({x}, {y})")
        
        if len(self.peg_points) == 2:
            self.calculate_beam_geometry()
            self.ui_mode = "ball_measurement"
    
    def calculate_beam_geometry(self):
        """Calculate beam geometry from peg points"""
        p1, p2 = self.peg_points
        
        self.beam_center_px = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        self.beam_vector_px = (p2[0] - p1[0], p2[1] - p1[1])
        self.beam_length_px = math.sqrt(self.beam_vector_px[0]**2 + self.beam_vector_px[1]**2)
        self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / self.beam_length_px
        
        print(f"[CAMERA] Beam center: {self.beam_center_px}")
        print(f"[CAMERA] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f}")
    
    def measure_ball_diameter(self):
        """Measure ball diameter"""
        if self.current_frame is None:
            return False
        
        found, center, radius, _ = self.ball_detector.detect_ball(self.current_frame)
        if not found:
            print("[BALL] Ball not detected")
            return False
        
        self.ball_measurements.append(radius)
        
        avg_radius = np.mean(self.ball_measurements)
        avg_diameter_px = 2 * avg_radius
        self.ball_pixel_to_meter = self.BALL_DIAMETER_M / avg_diameter_px
        
        print(f"[BALL] Measurement {len(self.ball_measurements)}: radius={radius:.1f}px")
        print(f"[BALL] Ball ratio: {self.ball_pixel_to_meter:.6f} m/px")
        
        return True
    
    def ball_position_along_beam(self, ball_center_px):
        """Convert ball center to position along beam in meters"""
        if not self.beam_center_px or not self.pixel_to_meter_ratio:
            return None
        
        delta_x = ball_center_px[0] - self.beam_center_px[0]
        delta_y = ball_center_px[1] - self.beam_center_px[1]
        
        distance_magnitude = math.sqrt(delta_x**2 + delta_y**2)
        pos_px = math.copysign(distance_magnitude, delta_x)
        pos_m = pos_px * self.pixel_to_meter_ratio
        
        return pos_m
    
    def automatic_range_finding(self):
        """Start automatic range finding in a separate thread"""
        if not self.servo.arduino:
            print("[RANGE] Servo not connected!")
            return False
        
        if self.range_finding_active:
            print("[RANGE] Range finding already in progress!")
            return False
        
        # Start range finding in separate thread
        self.range_finding_active = True
        self.range_finding_thread = threading.Thread(target=self._range_finding_worker, daemon=True)
        self.range_finding_thread.start()
        return True
    
    def _range_finding_worker(self):
        """Worker thread for automatic range finding"""
        try:
            print("[RANGE] Starting automatic range finding...")
            self.range_positions = []
            
            test_angles = [
                self.servo.neutral_angle + 15,  # Left tilt
                self.servo.neutral_angle,      # Neutral
                self.servo.neutral_angle - 15   # Right tilt
            ]
            
            for i, angle in enumerate(test_angles):
                if not self.range_finding_active:
                    break
                    
                print(f"[RANGE] Moving to {angle}° (step {i+1}/3)")
                self.servo.send_angle(angle)
                time.sleep(2.0)  # Wait for ball to settle
                
                # Collect ball positions for 2 seconds
                positions = []
                start_time = time.time()
                
                while time.time() - start_time < 2.0 and self.range_finding_active:
                    if self.current_frame is not None:
                        found, center, _, _ = self.ball_detector.detect_ball(self.current_frame)
                        if found:
                            pos_m = self.ball_position_along_beam(center)
                            if pos_m is not None:
                                positions.append(pos_m)
                    time.sleep(0.02)  # 50 Hz sampling
                
                if positions:
                    avg_position = np.mean(positions)
                    self.range_positions.append(avg_position)
                    print(f"[RANGE] Angle {angle}°: position = {avg_position:.4f}m ({len(positions)} samples)")
            
            # Return to neutral and calculate results
            self.servo.move_to_neutral()
            
            if len(self.range_positions) >= 2:
                self.position_min = min(self.range_positions)
                self.position_max = max(self.range_positions)
                self.usable_range_m = self.position_max - self.position_min
                print(f"[RANGE] Min: {self.position_min:.4f}m, Max: {self.position_max:.4f}m")
                self.ui_mode = "complete"
                self.calibration_complete = True
        
        finally:
            self.range_finding_active = False
        
    def draw_overlay(self, frame):
        """Draw calibration overlay"""
        overlay = frame.copy()
        
        # Ball detection
        found, center, radius, _ = self.ball_detector.detect_ball(frame)
        if found:
            color = (0, 255, 255) if self.ui_mode != "color_calibration" else (0, 255, 0)
            cv2.circle(overlay, center, int(radius), color, 2)
            cv2.circle(overlay, center, 3, color, -1)
            
            # Show position if calibrated
            if self.beam_center_px:
                pos_m = self.ball_position_along_beam(center)
                if pos_m is not None:
                    cv2.putText(overlay, f"Pos: {pos_m:.4f}m", 
                               (center[0]+20, center[1]+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw pegs and beam
        for i, peg in enumerate(self.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+15, peg[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(self.peg_points) == 2:
            cv2.line(overlay, self.peg_points[0], self.peg_points[1], (255, 0, 0), 3)
            cv2.circle(overlay, self.beam_center_px, 5, (255, 0, 0), -1)
        
        # UI info
        y = 30
        line_h = 25
        
        cv2.putText(overlay, f"Mode: {self.ui_mode}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, self.instructions[self.ui_mode], (10, y + line_h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show calibration data
        if self.ui_mode in ["color_calibration"]:
            cv2.putText(overlay, f"HSV Samples: {len(self.hsv_samples)}", 
                       (10, y + 2*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if len(self.hsv_samples) > 0:
                cv2.putText(overlay, f"Lower: {self.ball_detector.lower_bound}", 
                           (10, y + 3*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                cv2.putText(overlay, f"Upper: {self.ball_detector.upper_bound}", 
                           (10, y + 4*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        if self.pixel_to_meter_ratio:
            cv2.putText(overlay, f"Pixel ratio: {self.pixel_to_meter_ratio:.6f} m/px", 
                       (10, y + 5*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.ball_pixel_to_meter:
            cv2.putText(overlay, f"Ball ratio: {self.ball_pixel_to_meter:.6f} m/px", 
                       (10, y + 6*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.position_min is not None and self.position_max is not None:
            cv2.putText(overlay, f"Range: {self.position_min:.4f}m to {self.position_max:.4f}m", 
                       (10, y + 7*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        if self.range_finding_active:
            cv2.putText(overlay, "AUTOMATIC RANGE FINDING IN PROGRESS...", 
                        (10, y + 2*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(overlay, f"Positions collected: {len(self.range_positions)}", 
                        (10, y + 3*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        return overlay
    
    def save_config(self, filename="config.json"):
        """Save complete configuration to JSON"""
        config = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0",
            
            # Physical parameters
            "beam_length_m": self.BEAM_LENGTH_M,
            "ball_diameter_m": self.BALL_DIAMETER_M,
            
            # Camera settings
            "camera": {
                "index": self.CAM_INDEX,
                "frame_width": self.FRAME_W,
                "frame_height": self.FRAME_H
            },
            
            # Ball detection HSV bounds
            "ball_detection": {
                "lower_hsv": self.ball_detector.lower_bound.tolist(),
                "upper_hsv": self.ball_detector.upper_bound.tolist(),
                "hsv_samples_collected": len(self.hsv_samples)
            },
            
            # Camera calibration
            "camera_calibration": {
                "peg_points": self.peg_points,
                "beam_center_px": self.beam_center_px,
                "beam_vector_px": self.beam_vector_px,
                "beam_length_px": self.beam_length_px,
                "pixel_to_meter_ratio": self.pixel_to_meter_ratio,
                "ball_pixel_to_meter_ratio": self.ball_pixel_to_meter,
                "ball_measurements": [float(r) for r in self.ball_measurements]
            },
            
            # Operating range
            "operating_range": {
                "position_min_m": self.position_min,
                "position_max_m": self.position_max,
                "usable_range_m": self.usable_range_m,
                "range_positions": [float(p) for p in self.range_positions]
            },
            
            # Servo settings
            "servo": {
                "port": self.servo.port,
                "neutral_angle": self.servo.neutral_angle,
                "min_angle": self.servo.min_angle,
                "max_angle": self.servo.max_angle
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[SAVE] Complete configuration saved to {filename}")
        return filename
    
    def run_calibration(self):
        """Main calibration loop"""
        # Connect to servo
        if not self.servo.connect():
            response = input("Continue without servo? (y/n): ")
            if response.lower() != 'y':
                return
        
        # Setup camera
        cap = cv2.VideoCapture(self.CAM_INDEX,cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {self.CAM_INDEX}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cv2.namedWindow("Complete Ball Balancer Calibration")
        cv2.setMouseCallback("Complete Ball Balancer Calibration", self.mouse_callback)
        
        print("[INFO] Complete Ball Balancer Calibration")
        print("[WORKFLOW]")
        print("1. Color calibration - click on ball")
        print("2. Camera calibration - click on pegs") 
        print("3. Ball measurement - press SPACE")
        print("4. Automatic range finding - press 'a'")
        print("5. Save configuration - press 's'")
        print("[CONTROLS]")
        print("  Click: Sample colors or select pegs")
        print("  'c': Complete color calibration")
        print("  SPACE: Measure ball diameter")
        print("  'a': Automatic range finding")
        print("  's': Save config.json")
        print("  'q/ESC': Quit")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                self.current_frame = frame.copy()
                
                # Draw overlay and display
                display_frame = self.draw_overlay(frame)
                cv2.imshow("Complete Ball Balancer Calibration", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key in (ord('q'), 27):  # Quit
                    break
                elif key == ord('c'):  # Complete color calibration
                    if self.ui_mode == "color_calibration" and len(self.hsv_samples) > 0:
                        self.ui_mode = "camera_calibration"
                        print("[INFO] Color calibration complete. Now click on pegs.")
                elif key == ord(' '):  # Measure ball
                    if self.ui_mode == "ball_measurement":
                        if self.measure_ball_diameter():
                            if len(self.ball_measurements) >= 3:
                                self.ui_mode = "range_finding"
                                print("[INFO] Ball measurements complete. Press 'a' for automatic range finding.")
                elif key == ord('a'):  # Automatic range finding
                    if self.ui_mode == "range_finding":
                        self.automatic_range_finding()
                elif key == ord('s'):  # Save config
                    if self.calibration_complete:
                        self.save_config()
                    else:
                        print("[WARN] Complete all calibration steps first")
                elif key == ord('r'):  # Reset (hidden)
                    self.__init__()
                    if self.servo.arduino:
                        self.servo.connect()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.servo.disconnect()
        
        if self.calibration_complete:
            self.print_final_summary()
    
    def print_final_summary(self):
        """Print final calibration summary"""
        print("\n" + "="*70)
        print("COMPLETE CALIBRATION SUMMARY")
        print("="*70)
        print(f"HSV bounds: L{self.ball_detector.lower_bound} U{self.ball_detector.upper_bound}")
        print(f"Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f} m/px")
        print(f"Ball verification ratio: {self.ball_pixel_to_meter:.6f} m/px")
        print(f"Operating range: {self.position_min:.4f}m to {self.position_max:.4f}m")
        print(f"Usable range: {self.usable_range_m:.4f}m")
        print("\nConfig saved to config.json - ready for control system!")
        print("="*70)

def main():
    print("Complete Ball Balancer Calibration System")
    
    calibrator = ComprehensiveCalibrator()
    
    # Configure parameters
    try:
        beam_len = input(f"Beam length in meters (default {calibrator.BEAM_LENGTH_M}): ").strip()
        if beam_len:
            calibrator.BEAM_LENGTH_M = float(beam_len)
        
        ball_dia = input(f"Ball diameter in meters (default {calibrator.BALL_DIAMETER_M}): ").strip()
        if ball_dia:
            calibrator.BALL_DIAMETER_M = float(ball_dia)
            
        servo_port = input(f"Arduino port (default {calibrator.servo.port}): ").strip()
        if servo_port:
            calibrator.servo.port = servo_port
    except ValueError:
        print("Using default values")
    
    calibrator.run_calibration()

if __name__ == "__main__":
    main()