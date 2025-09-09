import cv2
import numpy as np
import time

class HSVCalibrator:
    def __init__(self, camera_index=0):
        # Use DirectShow backend (confirmed working from diagnostic)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            print("ERROR: Could not open camera!")
            return
            
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify we can read frames
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            print("ERROR: Camera opened but cannot read frames!")
            return
            
        print(f"Camera opened successfully with DirectShow backend!")
        print(f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        # Storage for sampled HSV values
        self.hsv_samples = []
        
        # Current bounds (start with wide range)
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])
        
        # UI state
        self.sample_radius = 3  # Radius around click point to sample
        
        print("HSV Ball Detection Calibrator")
        print("=" * 40)
        print("Instructions:")
        print("- Left click on the ball in the Camera Feed to sample HSV values")
        print("- Right click to clear all samples")
        print("- Watch the HSV Mask window update in real-time as you sample")
        print("- Press 'r' to reset bounds to full range")
        print("- Press 's' to save current bounds to console")
        print("- Press 'q' or ESC to quit")
        print("- Use +/- to adjust sample radius")
        print()
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.sample_hsv_at_point(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clear_samples()
    
    def sample_hsv_at_point(self, x, y):
        """Sample HSV values in a small region around the clicked point"""
        if not hasattr(self, 'current_hsv'):
            return
            
        h, w = self.current_hsv.shape[:2]
        
        # Define sampling region
        x1 = max(0, x - self.sample_radius)
        x2 = min(w, x + self.sample_radius + 1)
        y1 = max(0, y - self.sample_radius)
        y2 = min(h, y + self.sample_radius + 1)
        
        # Extract HSV values from the region
        region = self.current_hsv[y1:y2, x1:x2]
        region_flat = region.reshape(-1, 3)
        
        # Add all pixels in the region to samples
        for hsv_pixel in region_flat:
            self.hsv_samples.append(hsv_pixel.copy())
        
        print(f"Sampled {len(region_flat)} pixels at ({x}, {y})")
        print(f"Total samples: {len(self.hsv_samples)}")
        
        # Update bounds
        self.update_bounds()
    
    def update_bounds(self):
        """Calculate optimal upper and lower bounds from all samples"""
        if not self.hsv_samples:
            return
            
        samples_array = np.array(self.hsv_samples)
        
        # Calculate bounds with some margin
        h_values = samples_array[:, 0]
        s_values = samples_array[:, 1]
        v_values = samples_array[:, 2]
        
        # For hue, handle the circular nature (0-179)
        h_min, h_max = np.min(h_values), np.max(h_values)
        h_range = h_max - h_min
        
        # Add margin (but be careful with hue wraparound)
        h_margin = max(5, h_range * 0.1)
        s_margin = max(10, (np.max(s_values) - np.min(s_values)) * 0.15)
        v_margin = max(10, (np.max(v_values) - np.min(v_values)) * 0.15)
        
        # Calculate bounds
        self.lower_bound[0] = max(0, h_min - h_margin)
        self.upper_bound[0] = min(179, h_max + h_margin)
        
        self.lower_bound[1] = max(0, np.min(s_values) - s_margin)
        self.upper_bound[1] = min(255, np.max(s_values) + s_margin)
        
        self.lower_bound[2] = max(0, np.min(v_values) - v_margin)
        self.upper_bound[2] = min(255, np.max(v_values) + v_margin)
        
        print(f"Updated bounds:")
        print(f"  Lower: H={self.lower_bound[0]:3.0f} S={self.lower_bound[1]:3.0f} V={self.lower_bound[2]:3.0f}")
        print(f"  Upper: H={self.upper_bound[0]:3.0f} S={self.upper_bound[1]:3.0f} V={self.upper_bound[2]:3.0f}")
        print()
    
    def clear_samples(self):
        """Clear all samples and reset bounds"""
        self.hsv_samples = []
        self.lower_bound = np.array([0, 0, 0])
        self.upper_bound = np.array([179, 255, 255])
        print("Cleared all samples and reset bounds")
        print()
    
    def create_mask(self, hsv_image):
        """Create mask using current bounds"""
        mask = cv2.inRange(hsv_image, self.lower_bound, self.upper_bound)
        
        # Clean up the mask
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=2)
        
        return mask
    
    def draw_ui(self, frame):
        """Draw UI elements on the frame"""
        h, w = frame.shape[:2]
        
        # Create info panel
        info_height = 120
        info_panel = np.zeros((info_height, w, 3), dtype=np.uint8)
        
        # Sample count
        cv2.putText(info_panel, f"Samples: {len(self.hsv_samples)}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Current bounds
        cv2.putText(info_panel, f"Lower HSV: [{self.lower_bound[0]:3.0f}, {self.lower_bound[1]:3.0f}, {self.lower_bound[2]:3.0f}]", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(info_panel, f"Upper HSV: [{self.upper_bound[0]:3.0f}, {self.upper_bound[1]:3.0f}, {self.upper_bound[2]:3.0f}]", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Sample radius
        cv2.putText(info_panel, f"Sample radius: {self.sample_radius} (use +/- to adjust)", 
                   (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Combine frame and info panel
        combined = np.vstack([frame, info_panel])
        return combined
    
    def save_bounds_to_console(self):
        """Print the current bounds in a format ready for copy-paste"""
        print("\n" + "="*50)
        print("COPY THESE VALUES TO YOUR BALL_DETECTION.PY:")
        print("="*50)
        print(f"lower_orange = np.array([{self.lower_bound[0]:.0f}, {self.lower_bound[1]:.0f}, {self.lower_bound[2]:.0f}])")
        print(f"upper_orange = np.array([{self.upper_bound[0]:.0f}, {self.upper_bound[1]:.0f}, {self.upper_bound[2]:.0f}])")
        print("="*50)
        print()
    
    def run(self):
        """Main calibration loop"""
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set up two windows
        cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("HSV Mask", cv2.WINDOW_AUTOSIZE)
        
        # Position windows side by side
        cv2.moveWindow("Camera Feed", 100, 100)
        cv2.moveWindow("HSV Mask", 750, 100)
        
        # Set up mouse callback for the camera feed window
        cv2.setMouseCallback("Camera Feed", self.mouse_callback)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            self.current_hsv = hsv  # Store for mouse callback
            
            # Create and show camera feed with UI
            camera_display = self.draw_ui(frame.copy())
            cv2.imshow("Camera Feed", camera_display)
            
            # Create and show mask
            mask = self.create_mask(hsv)
            cv2.imshow("HSV Mask", mask)
            
            # Handle keyboard input (from either window)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                break
            elif key == ord('r'):
                self.clear_samples()
            elif key == ord('s'):
                self.save_bounds_to_console()
            elif key == ord('+') or key == ord('='):
                self.sample_radius = min(10, self.sample_radius + 1)
                print(f"Sample radius: {self.sample_radius}")
            elif key == ord('-'):
                self.sample_radius = max(1, self.sample_radius - 1)
                print(f"Sample radius: {self.sample_radius}")
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Final output
        if self.hsv_samples:
            self.save_bounds_to_console()

def main():
    calibrator = HSVCalibrator(camera_index=0)
    calibrator.run()

if __name__ == "__main__":
    main()