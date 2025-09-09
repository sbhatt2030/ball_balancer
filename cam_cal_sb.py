"""
Practical Camera Calibration for Ball Balancer

Calibration Process:
1. Click on 2 peg endpoints to establish pixel-to-meter ratio
2. Use ball diameter for secondary verification  
3. Record ball movement to find actual min/max usable positions
4. Export calibration data for use in control system

This gives you the REAL usable range, not theoretical beam length.
"""

import cv2
import numpy as np
import time
import json
import os
from datetime import datetime
import math
from collections import deque

try:
    from ball_detection import detect_ball_x
except Exception as e:
    print("[ERROR] Could not import detect_ball_x from ball_detection.py:", e)
    raise

class PracticalCalibrator:
    def __init__(self):
        # Physical parameters (USER CONFIGURABLE)
        self.BEAM_LENGTH_M = 0.20      # Total beam length in meters
        self.BALL_DIAMETER_M = 0.025   # Ball diameter in meters
        
        # Camera settings
        self.CAM_INDEX = 0
        self.FRAME_W, self.FRAME_H = 640, 480
        
        # Calibration state
        self.peg_points = []           # (x, y) pixel coordinates for pegs
        self.beam_center_px = None     # Calculated beam center
        self.beam_vector_px = None     # Vector along beam direction
        self.beam_length_px = None     # Beam length in pixels
        self.pixel_to_meter_ratio = None  # Primary calibration ratio
        
        # Ball diameter verification
        self.ball_measurements = []    # List of ball diameter measurements
        self.ball_pixel_to_meter = None  # Secondary calibration from ball
        
        # RECORDING PHASE - Track actual ball movement
        self.recording = False
        self.recorded_positions = deque(maxlen=1000)  # Store last 1000 ball positions
        self.position_min = None       # Actual minimum ball position reached
        self.position_max = None       # Actual maximum ball position reached
        self.recording_start_time = None
        
        # Current frame
        self.current_frame = None
        
        # UI state management
        self.ui_mode = "select_pegs"   # "select_pegs", "measure_ball", "record_range", "complete"
        self.instructions = {
            "select_pegs": "Click on the TWO PEGS that define the beam endpoints",
            "measure_ball": "Place ball on beam. Press SPACE to measure ball diameter (3-5 times)",
            "record_range": "Press 'r' to START recording, move ball back and forth, press 'd' when DONE",
            "complete": "Calibration complete! Min/Max positions recorded. Press 's' to save"
        }
        
        # Results
        self.calibration_complete = False
        self.usable_range_m = None     # Actual usable range in meters
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for peg selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.ui_mode == "select_pegs" and len(self.peg_points) < 2:
                self.peg_points.append((x, y))
                print(f"[PEG] Selected peg {len(self.peg_points)} at ({x}, {y})")
                
                if len(self.peg_points) == 2:
                    self.calculate_beam_geometry()
                    self.ui_mode = "measure_ball"
    
    def calculate_beam_geometry(self):
        """Calculate beam geometry from peg positions"""
        if len(self.peg_points) != 2:
            return
        
        p1, p2 = self.peg_points
        
        # Calculate beam center (midpoint)
        self.beam_center_px = (
            (p1[0] + p2[0]) // 2,
            (p1[1] + p2[1]) // 2
        )
        
        # Calculate beam vector and length
        self.beam_vector_px = (p2[0] - p1[0], p2[1] - p1[1])
        self.beam_length_px = math.sqrt(self.beam_vector_px[0]**2 + self.beam_vector_px[1]**2)
        
        # Calculate primary pixel-to-meter ratio
        self.pixel_to_meter_ratio = self.BEAM_LENGTH_M / self.beam_length_px
        
        print(f"[GEOMETRY] Beam center: {self.beam_center_px}")
        print(f"[GEOMETRY] Beam length: {self.beam_length_px:.1f} pixels = {self.BEAM_LENGTH_M}m")
        print(f"[GEOMETRY] Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f} m/pixel")
    
    def detect_ball_details(self, frame):
        """Detect ball and return center and radius"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Orange detection (same as ball_detection.py)
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([20, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        largest_contour = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
        
        if radius < 5 or radius > 100:
            return None, None
        
        return (int(x), int(y)), radius
    
    def measure_ball_diameter(self):
        """Measure ball diameter for verification"""
        if self.current_frame is None:
            return False
        
        ball_center_px, ball_radius_px = self.detect_ball_details(self.current_frame)
        if ball_center_px is None:
            print("[WARN] Ball not detected for diameter measurement")
            return False
        
        self.ball_measurements.append(ball_radius_px)
        
        # Calculate average and ratio
        avg_radius = np.mean(self.ball_measurements)
        avg_diameter_px = 2 * avg_radius
        ball_ratio = self.BALL_DIAMETER_M / avg_diameter_px
        
        print(f"[BALL] Measurement {len(self.ball_measurements)}: radius={ball_radius_px:.1f}px")
        print(f"[BALL] Average diameter: {avg_diameter_px:.1f}px → ratio: {ball_ratio:.6f} m/px")
        
        # Update ball calibration
        self.ball_pixel_to_meter = ball_ratio
        
        # Compare ratios
        if self.pixel_to_meter_ratio:
            diff = abs(self.pixel_to_meter_ratio - ball_ratio)
            error_pct = 100 * diff / self.pixel_to_meter_ratio
            print(f"[VERIFICATION] Ratio difference: {error_pct:.1f}%")
        
        return True
    
    def ball_position_along_beam(self, ball_center_px):
        """Convert ball center to position along beam in meters"""
        if not self.beam_center_px or not self.pixel_to_meter_ratio:
            return None
        
        # Vector from beam center to ball
        to_ball = (ball_center_px[0] - self.beam_center_px[0], 
                  ball_center_px[1] - self.beam_center_px[1])
        
        # Unit vector along beam
        beam_length = math.sqrt(self.beam_vector_px[0]**2 + self.beam_vector_px[1]**2)
        beam_unit = (self.beam_vector_px[0] / beam_length, self.beam_vector_px[1] / beam_length)
        
        # Project ball position onto beam direction (signed distance)
        distance_px = to_ball[0] * beam_unit[0] + to_ball[1] * beam_unit[1]
        distance_m = distance_px * self.pixel_to_meter_ratio
        
        return distance_m
    
    def start_recording(self):
        """Start recording ball positions"""
        self.recording = True
        self.recorded_positions.clear()
        self.position_min = None
        self.position_max = None
        self.recording_start_time = time.time()
        print("[RECORD] Started recording ball positions. Move ball back and forth!")
    
    def stop_recording(self):
        """Stop recording and calculate min/max"""
        self.recording = False
        
        if len(self.recorded_positions) == 0:
            print("[WARN] No positions recorded!")
            return
        
        # Calculate actual min/max from recorded data
        positions = np.array(self.recorded_positions)
        self.position_min = np.min(positions)
        self.position_max = np.max(positions)
        self.usable_range_m = self.position_max - self.position_min
        
        print(f"[RECORD] Stopped recording. {len(self.recorded_positions)} positions captured.")
        print(f"[RANGE] Minimum position: {self.position_min:.4f}m")
        print(f"[RANGE] Maximum position: {self.position_max:.4f}m") 
        print(f"[RANGE] Usable range: {self.usable_range_m:.4f}m")
        
        # Move to completion
        self.ui_mode = "complete"
        self.calibration_complete = True
    
    def update_recording(self):
        """Update recording with current ball position"""
        if not self.recording or self.current_frame is None:
            return
        
        ball_center, _ = self.detect_ball_details(self.current_frame)
        if ball_center:
            position_m = self.ball_position_along_beam(ball_center)
            if position_m is not None:
                self.recorded_positions.append(position_m)
    
    def draw_overlay(self, frame):
        """Draw calibration overlay"""
        overlay = frame.copy()
        
        # Draw pegs and beam line
        for i, peg in enumerate(self.peg_points):
            cv2.circle(overlay, peg, 8, (0, 255, 0), -1)
            cv2.putText(overlay, f"Peg {i+1}", (peg[0]+15, peg[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(self.peg_points) == 2:
            cv2.line(overlay, self.peg_points[0], self.peg_points[1], (255, 0, 0), 3)
            cv2.circle(overlay, self.beam_center_px, 5, (255, 0, 0), -1)
            cv2.putText(overlay, "Center", (self.beam_center_px[0]+10, self.beam_center_px[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw current ball detection
        ball_center, ball_radius = self.detect_ball_details(frame)
        if ball_center:
            # Ball circle
            color = (0, 255, 255) if not self.recording else (0, 255, 0)
            cv2.circle(overlay, ball_center, int(ball_radius), color, 2)
            cv2.circle(overlay, ball_center, 3, color, -1)
            
            # Ball position
            if self.beam_center_px:
                pos_m = self.ball_position_along_beam(ball_center)
                if pos_m is not None:
                    cv2.putText(overlay, f"Pos: {pos_m:.4f}m", 
                               (ball_center[0]+20, ball_center[1]+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # UI Information
        y = 30
        line_h = 25
        
        # Mode and instructions
        cv2.putText(overlay, f"Mode: {self.ui_mode}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(overlay, self.instructions[self.ui_mode], (10, y + line_h),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calibration info
        if self.pixel_to_meter_ratio:
            cv2.putText(overlay, f"Beam ratio: {self.pixel_to_meter_ratio:.6f} m/px", 
                       (10, y + 2*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        if self.ball_pixel_to_meter:
            cv2.putText(overlay, f"Ball ratio: {self.ball_pixel_to_meter:.6f} m/px ({len(self.ball_measurements)} samples)", 
                       (10, y + 3*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Recording info
        if self.recording:
            elapsed = time.time() - self.recording_start_time
            cv2.putText(overlay, f"RECORDING: {len(self.recorded_positions)} positions ({elapsed:.1f}s)", 
                       (10, y + 4*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Range results
        if self.position_min is not None and self.position_max is not None:
            cv2.putText(overlay, f"Range: {self.position_min:.4f}m to {self.position_max:.4f}m ({self.usable_range_m:.4f}m)", 
                       (10, y + 5*line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        return overlay
    
    def save_calibration(self, filename=None):
        """Save complete calibration including min/max range"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"practical_calibration_{timestamp}.json"
        
        calibration_data = {
            'timestamp': datetime.now().isoformat(),
            'beam_length_m': self.BEAM_LENGTH_M,
            'ball_diameter_m': self.BALL_DIAMETER_M,
            'frame_size': [self.FRAME_W, self.FRAME_H],
            
            # Geometric calibration
            'peg_points': self.peg_points,
            'beam_center_px': self.beam_center_px,
            'beam_vector_px': self.beam_vector_px,
            'beam_length_px': self.beam_length_px,
            'pixel_to_meter_ratio': self.pixel_to_meter_ratio,
            'ball_pixel_to_meter': self.ball_pixel_to_meter,
            'ball_measurements': [float(r) for r in self.ball_measurements],
            
            # ACTUAL USABLE RANGE (the key results!)
            'position_min_m': self.position_min,
            'position_max_m': self.position_max,
            'usable_range_m': self.usable_range_m,
            'recorded_positions_count': len(self.recorded_positions),
            'calibration_complete': self.calibration_complete
        }
        
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"[SAVE] Calibration saved to: {filename}")
        
        # Also save a simple config for your control system
        config_file = filename.replace('.json', '_config.json')
        simple_config = {
            'scale_factor': self.pixel_to_meter_ratio,
            'beam_center_px': self.beam_center_px,
            'beam_vector_px': self.beam_vector_px,
            'position_min_m': self.position_min,
            'position_max_m': self.position_max,
            'usable_range_m': self.usable_range_m
        }
        
        with open(config_file, 'w') as f:
            json.dump(simple_config, f, indent=2)
        
        print(f"[SAVE] Simple config saved to: {config_file}")
        return filename
    
    def run_calibration(self):
        """Main calibration loop"""
        cap = cv2.VideoCapture(self.CAM_INDEX,cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[ERROR] Could not open camera {self.CAM_INDEX}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        cv2.namedWindow("Practical Ball Balancer Calibration")
        cv2.setMouseCallback("Practical Ball Balancer Calibration", self.mouse_callback)
        
        print("[INFO] Practical Ball Balancer Calibration")
        print(f"[INFO] Beam: {self.BEAM_LENGTH_M}m, Ball: {self.BALL_DIAMETER_M}m")
        print("\n[WORKFLOW]")
        print("1. Click on 2 peg endpoints")
        print("2. Measure ball diameter (SPACE key, 3-5 times)")
        print("3. Record actual range ('r' to start, move ball, 'd' when done)")
        print("4. Save results ('s' key)")
        print("\n[CONTROLS]")
        print("  Click: Select pegs")
        print("  SPACE: Measure ball diameter")  
        print("  'r': Start recording range")
        print("  'd': Done recording")
        print("  's': Save calibration")
        print("  'q/ESC': Quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            self.current_frame = frame.copy()
            
            # Update recording if active
            self.update_recording()
            
            # Draw and display
            display_frame = self.draw_overlay(frame)
            cv2.imshow("Practical Ball Balancer Calibration", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord('q'), 27):  # Quit
                break
            elif key == ord(' '):  # Measure ball diameter
                if self.ui_mode == "measure_ball":
                    if self.measure_ball_diameter():
                        if len(self.ball_measurements) >= 3:
                            self.ui_mode = "record_range"
                            print("[INFO] Ball measurements complete. Ready to record range.")
            elif key == ord('r'):  # Start recording
                if self.ui_mode == "record_range" and not self.recording:
                    self.start_recording()
            elif key == ord('d'):  # Done recording
                if self.recording:
                    self.stop_recording()
            elif key == ord('s'):  # Save
                if self.calibration_complete:
                    self.save_calibration()
                else:
                    print("[WARN] Complete all calibration steps first")
            elif key == ord('x'):  # Reset (hidden feature)
                self.__init__()
                print("[INFO] Calibration reset")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if self.calibration_complete:
            self.print_final_summary()
    
    def print_final_summary(self):
        """Print comprehensive calibration results"""
        print("\n" + "="*70)
        print("PRACTICAL CALIBRATION RESULTS")
        print("="*70)
        print(f"Beam length: {self.BEAM_LENGTH_M}m ({self.beam_length_px:.1f} pixels)")
        print(f"Pixel-to-meter ratio: {self.pixel_to_meter_ratio:.6f} m/pixel")
        
        if self.ball_pixel_to_meter:
            diff = abs(self.pixel_to_meter_ratio - self.ball_pixel_to_meter)
            error_pct = 100 * diff / self.pixel_to_meter_ratio
            print(f"Ball verification: {error_pct:.1f}% difference ({'✓ GOOD' if error_pct < 10 else '⚠ CHECK'})")
        
        print("\n** ACTUAL USABLE RANGE (Key Results) **")
        print(f"Minimum position: {self.position_min:.4f}m")
        print(f"Maximum position: {self.position_max:.4f}m")
        print(f"Total usable range: {self.usable_range_m:.4f}m")
        print(f"Range utilization: {100*self.usable_range_m/self.BEAM_LENGTH_M:.1f}% of beam length")
        
        print(f"\nRecorded positions: {len(self.recorded_positions)}")
        print("\nUse these min/max values in your control system!")
        print("="*70 + "\n")

def main():
    print("Practical Ball Balancer Calibration")
    
    calibrator = PracticalCalibrator()
    
    # Configure parameters
    try:
        beam_len = input(f"Beam length in meters (default {calibrator.BEAM_LENGTH_M}): ").strip()
        if beam_len:
            calibrator.BEAM_LENGTH_M = float(beam_len)
        
        ball_dia = input(f"Ball diameter in meters (default {calibrator.BALL_DIAMETER_M}): ").strip()
        if ball_dia:
            calibrator.BALL_DIAMETER_M = float(ball_dia)
    except ValueError:
        print("Using default values")
    
    calibrator.run_calibration()

if __name__ == "__main__":
    main()