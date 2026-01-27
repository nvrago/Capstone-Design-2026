#!/usr/bin/env python3
"""
Calibration Script

Camera calibration, laser plane fitting, and HSV threshold tuning
for software-only laser detection (no physical red filter needed).
"""

import argparse
import logging
import sys
import time
import yaml
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from scanner.camera import Camera, CameraConfig
from scanner.laser import LaserDetector, LaserConfig

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LaserCalibrator:
    """
    Software-based laser detection calibration.
    
    Uses background subtraction and HSV tuning to detect laser
    without physical filters.
    """
    
    def __init__(self, camera: Camera):
        self.camera = camera
        self.background = None
        self.hsv_lower = np.array([0, 100, 100])
        self.hsv_upper = np.array([10, 255, 255])
        self.hsv_lower_alt = np.array([170, 100, 100])
        self.hsv_upper_alt = np.array([180, 255, 255])
        
    def capture_background(self, num_frames: int = 5) -> np.ndarray:
        """
        Capture background image with laser OFF.
        
        Call this before turning on the laser.
        """
        print("\n=== BACKGROUND CAPTURE ===")
        print("Ensure laser is OFF, then press Enter...")
        input()
        
        frames = []
        for i in range(num_frames):
            frame = self.camera.capture()
            if frame is not None:
                frames.append(frame.astype(np.float32))
            time.sleep(0.1)
            
        self.background = np.median(frames, axis=0).astype(np.uint8)
        logger.info(f"Captured background from {len(frames)} frames")
        return self.background
        
    def subtract_background(self, image: np.ndarray) -> np.ndarray:
        """
        Subtract background to isolate laser.
        
        Returns difference image highlighting laser pixels.
        """
        if self.background is None:
            return image
            
        # Absolute difference
        diff = cv2.absdiff(image, self.background)
        
        # Convert to grayscale and threshold
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Amplify the difference
        enhanced = cv2.convertScaleAbs(gray_diff, alpha=3.0, beta=0)
        
        return enhanced
        
    def detect_with_subtraction(self, image: np.ndarray) -> np.ndarray:
        """
        Detect laser using background subtraction + HSV filtering.
        
        More robust than HSV alone in varying lighting.
        """
        # Background subtraction
        if self.background is not None:
            diff = cv2.absdiff(image, self.background)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
        else:
            motion_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
            
        # HSV filtering for red
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask2 = cv2.inRange(hsv, self.hsv_lower_alt, self.hsv_upper_alt)
        color_mask = cv2.bitwise_or(mask1, mask2)
        
        # Combine: must be both changed AND red
        combined = cv2.bitwise_and(color_mask, motion_mask)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and extract centroids
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        points = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 5 < area < 1000:
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = M['m10'] / M['m00']
                    cy = M['m01'] / M['m00']
                    points.append([cx, cy])
                    
        return np.array(points, dtype=np.float32).reshape(-1, 2)
        
    def auto_tune_hsv(self, num_samples: int = 10) -> dict:
        """
        Automatically determine HSV thresholds for laser detection.
        
        Captures frames with laser ON and analyzes brightest pixels.
        """
        print("\n=== AUTO HSV TUNING ===")
        print("Turn laser ON and aim at a surface, then press Enter...")
        input()
        
        all_h, all_s, all_v = [], [], []
        
        for i in range(num_samples):
            frame = self.camera.capture()
            if frame is None:
                continue
                
            # Use background subtraction to find laser pixels
            if self.background is not None:
                diff = cv2.absdiff(frame, self.background)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                # Find significantly changed pixels
                bright_mask = diff_gray > np.percentile(diff_gray, 98)
            else:
                # Fallback: just use brightest pixels
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bright_mask = gray > np.percentile(gray, 99)
                
            if bright_mask.sum() < 10:
                continue
                
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            all_h.extend(hsv[:,:,0][bright_mask].tolist())
            all_s.extend(hsv[:,:,1][bright_mask].tolist())
            all_v.extend(hsv[:,:,2][bright_mask].tolist())
            
            time.sleep(0.1)
            
        if len(all_h) < 50:
            logger.warning("Not enough laser pixels detected for tuning")
            return None
            
        # Analyze hue distribution (handle red wraparound at 0/180)
        h_arr = np.array(all_h)
        
        # Check if hue wraps around (red laser)
        low_hue = (h_arr < 20).sum()
        high_hue = (h_arr > 160).sum()
        
        if low_hue > 0 and high_hue > 0:
            # Red wraparound case
            h_low = h_arr[h_arr < 90]
            h_high = h_arr[h_arr >= 90]
            
            result = {
                'hsv_lower': [0, int(np.percentile(all_s, 5)), int(np.percentile(all_v, 5))],
                'hsv_upper': [int(np.percentile(h_low, 95)) if len(h_low) > 0 else 10, 255, 255],
                'hsv_lower_alt': [int(np.percentile(h_high, 5)) if len(h_high) > 0 else 170, 
                                  int(np.percentile(all_s, 5)), int(np.percentile(all_v, 5))],
                'hsv_upper_alt': [180, 255, 255],
                'samples': len(all_h)
            }
        else:
            # Non-wrapping case
            result = {
                'hsv_lower': [int(np.percentile(all_h, 5)), 
                             int(np.percentile(all_s, 5)), 
                             int(np.percentile(all_v, 5))],
                'hsv_upper': [int(np.percentile(all_h, 95)), 255, 255],
                'hsv_lower_alt': [0, 0, 0],  # unused
                'hsv_upper_alt': [0, 0, 0],  # unused
                'samples': len(all_h)
            }
            
        logger.info(f"Auto-tuned HSV from {result['samples']} pixels")
        logger.info(f"  Lower: {result['hsv_lower']}, Upper: {result['hsv_upper']}")
        
        # Update internal state
        self.hsv_lower = np.array(result['hsv_lower'])
        self.hsv_upper = np.array(result['hsv_upper'])
        self.hsv_lower_alt = np.array(result['hsv_lower_alt'])
        self.hsv_upper_alt = np.array(result['hsv_upper_alt'])
        
        return result
        
    def interactive_tune(self):
        """
        Interactive HSV tuning with live preview.
        
        Opens window with trackbars for real-time adjustment.
        """
        print("\n=== INTERACTIVE HSV TUNING ===")
        print("Turn laser ON. Adjust sliders until only laser is visible.")
        print("Press 'q' to quit, 's' to save current values.")
        
        cv2.namedWindow('Tuning')
        cv2.createTrackbar('H_low', 'Tuning', 0, 180, lambda x: None)
        cv2.createTrackbar('H_high', 'Tuning', 10, 180, lambda x: None)
        cv2.createTrackbar('S_low', 'Tuning', 100, 255, lambda x: None)
        cv2.createTrackbar('V_low', 'Tuning', 100, 255, lambda x: None)
        
        saved_values = None
        
        while True:
            frame = self.camera.capture()
            if frame is None:
                continue
                
            h_low = cv2.getTrackbarPos('H_low', 'Tuning')
            h_high = cv2.getTrackbarPos('H_high', 'Tuning')
            s_low = cv2.getTrackbarPos('S_low', 'Tuning')
            v_low = cv2.getTrackbarPos('V_low', 'Tuning')
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Primary mask
            mask = cv2.inRange(hsv, 
                              np.array([h_low, s_low, v_low]),
                              np.array([h_high, 255, 255]))
                              
            # Red wraparound
            if h_low < 20:
                mask2 = cv2.inRange(hsv,
                                   np.array([180 - h_high, s_low, v_low]),
                                   np.array([180, 255, 255]))
                mask = cv2.bitwise_or(mask, mask2)
                
            # Background subtraction overlay
            if self.background is not None:
                diff = self.subtract_background(frame)
                diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            else:
                diff_color = np.zeros_like(frame)
                
            # Display
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Count detected pixels
            pixel_count = cv2.countNonZero(mask)
            cv2.putText(result, f"Pixels: {pixel_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                       
            display = np.hstack([frame, result])
            display = cv2.resize(display, (1280, 360))
            cv2.imshow('Tuning', display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                saved_values = {
                    'hsv_lower': [h_low, s_low, v_low],
                    'hsv_upper': [h_high, 255, 255],
                    'hsv_lower_alt': [180 - h_high, s_low, v_low],
                    'hsv_upper_alt': [180, 255, 255]
                }
                logger.info(f"Saved: {saved_values}")
                
        cv2.destroyAllWindows()
        return saved_values


def calibrate_camera(camera: Camera, output_dir: Path, 
                     board_size: tuple = (9, 6), square_size: float = 25.0):
    """
    Calibrate camera using checkerboard pattern.
    
    Args:
        board_size: Internal corners (width, height)
        square_size: Size of checkerboard squares in mm
    """
    print("\n=== CAMERA CALIBRATION ===")
    print(f"Using {board_size[0]}x{board_size[1]} checkerboard, {square_size}mm squares")
    print("Press SPACE to capture, 'c' to calibrate when done, 'q' to quit")
    
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    obj_points = []
    img_points = []
    
    cv2.namedWindow('Calibration')
    
    while True:
        frame = camera.capture()
        if frame is None:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        
        display = frame.copy()
        if ret:
            cv2.drawChessboardCorners(display, board_size, corners, ret)
            
        cv2.putText(display, f"Captures: {len(obj_points)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                   
        cv2.imshow('Calibration', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and ret:
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            obj_points.append(objp)
            img_points.append(corners_refined)
            logger.info(f"Captured frame {len(obj_points)}")
        elif key == ord('c') and len(obj_points) >= 5:
            break
            
    cv2.destroyAllWindows()
    
    if len(obj_points) < 5:
        logger.error("Need at least 5 captures for calibration")
        return None
        
    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    
    if ret:
        logger.info(f"Calibration successful. RMS error: {ret:.4f}")
        
        result = {
            'camera_matrix': mtx.tolist(),
            'dist_coeffs': dist.tolist(),
            'rms_error': ret,
            'resolution': list(gray.shape[::-1])
        }
        
        # Save
        output_path = output_dir / 'camera_calibration.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(result, f)
        logger.info(f"Saved calibration to {output_path}")
        
        return result
    else:
        logger.error("Calibration failed")
        return None


def main():
    parser = argparse.ArgumentParser(description='Scanner calibration tools')
    parser.add_argument('--camera', action='store_true', 
                        help='Calibrate camera intrinsics')
    parser.add_argument('--laser', action='store_true',
                        help='Calibrate laser detection (HSV tuning)')
    parser.add_argument('--auto', action='store_true',
                        help='Use automatic HSV tuning')
    parser.add_argument('--interactive', action='store_true',
                        help='Use interactive HSV tuning')
    parser.add_argument('--output', '-o', type=str, default='config',
                        help='Output directory')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize camera
    camera = Camera(CameraConfig(width=1920, height=1080))
    if not camera.initialize():
        logger.error("Failed to initialize camera")
        return 1
        
    try:
        if args.camera:
            calibrate_camera(camera, output_dir)
            
        if args.laser:
            calibrator = LaserCalibrator(camera)
            
            # Always capture background first
            calibrator.capture_background()
            
            if args.auto:
                result = calibrator.auto_tune_hsv()
            elif args.interactive:
                result = calibrator.interactive_tune()
            else:
                # Default: auto then verify interactively
                result = calibrator.auto_tune_hsv()
                print("\nVerify with interactive tuning? (y/n)")
                if input().lower() == 'y':
                    result = calibrator.interactive_tune() or result
                    
            if result:
                output_path = output_dir / 'laser_calibration.yaml'
                with open(output_path, 'w') as f:
                    yaml.dump(result, f)
                logger.info(f"Saved laser calibration to {output_path}")
                
        if not (args.camera or args.laser):
            parser.print_help()
            
    finally:
        camera.close()
        
    return 0


if __name__ == '__main__':
    sys.exit(main())