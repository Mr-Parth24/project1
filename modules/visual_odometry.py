"""
Visual Odometry Module
Estimates camera movement using feature tracking
"""

import cv2
import numpy as np
from collections import deque

class VisualOdometry:
    def __init__(self):
        # Feature detector (SIFT works better than ORB for this application)
        self.detector = cv2.SIFT_create(nfeatures=1000)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        # Current features for visualization
        self.current_features = []
        
        # Movement history for smoothing
        self.movement_history = deque(maxlen=5)
        
        # Camera matrix (approximate values for RealSense D435i)
        self.camera_matrix = np.array([
            [615.0, 0.0, 320.0],
            [0.0, 615.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        
        # Distance coefficients (none for simplicity)
        self.dist_coeffs = np.zeros((4, 1))
        
        # Minimum number of features required
        self.min_features = 50
        
        print("🔍 Visual Odometry initialized")
        
    def process_frame(self, color_frame, depth_frame):
        """Process a frame and return estimated movement"""
        # Convert to grayscale
        gray = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) < self.min_features:
            print(f"⚠️  Not enough features detected: {len(keypoints) if keypoints else 0}")
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return None
            
        # Store current features for visualization
        self.current_features = [kp.pt for kp in keypoints[:100]]  # Limit for performance
        
        movement = None
        
        # If we have previous frame, estimate movement
        if self.prev_descriptors is not None and len(self.prev_keypoints) >= self.min_features:
            movement = self.estimate_movement(
                self.prev_keypoints, self.prev_descriptors,
                keypoints, descriptors,
                depth_frame
            )
            
        # Update previous frame data
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        return movement
        
    def estimate_movement(self, prev_kp, prev_desc, curr_kp, curr_desc, depth_frame):
        """Estimate movement between two frames"""
        try:
            # Match features
            matches = self.matcher.match(prev_desc, curr_desc)
            
            if len(matches) < 20:
                print(f"⚠️  Not enough matches: {len(matches)}")
                return None
                
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Get good matches (top 70%)
            good_matches = matches[:int(len(matches) * 0.7)]
            
            if len(good_matches) < 15:
                return None
                
            # Extract matched points
            prev_pts = np.array([prev_kp[m.queryIdx].pt for m in good_matches])
            curr_pts = np.array([curr_kp[m.trainIdx].pt for m in good_matches])
            
            # Get 3D points using depth
            prev_3d = self.get_3d_points(prev_pts, depth_frame)
            curr_3d = self.get_3d_points(curr_pts, depth_frame)
            
            # Filter out invalid 3D points
            valid_mask = (prev_3d[:, 2] > 0) & (curr_3d[:, 2] > 0) & (prev_3d[:, 2] < 10) & (curr_3d[:, 2] < 10)
            
            if np.sum(valid_mask) < 10:
                return None
                
            prev_3d = prev_3d[valid_mask]
            curr_3d = curr_3d[valid_mask]
            
            # Estimate transformation using RANSAC
            movement = self.estimate_transform_ransac(prev_3d, curr_3d)
            
            if movement is not None:
                # Smooth movement
                self.movement_history.append(movement)
                if len(self.movement_history) >= 3:
                    # Use median filtering to reduce noise
                    movements = np.array(list(self.movement_history))
                    movement = np.median(movements, axis=0)
                    
            return movement
            
        except Exception as e:
            print(f"❌ Error in movement estimation: {e}")
            return None
            
    def get_3d_points(self, points_2d, depth_frame):
        """Convert 2D points to 3D using depth information"""
        points_3d = []
        
        for pt in points_2d:
            x, y = int(pt[0]), int(pt[1])
            
            # Check bounds
            if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0]:
                # Get depth (in mm, convert to meters)
                depth = depth_frame[y, x] / 1000.0
                
                if depth > 0:
                    # Convert to 3D using camera intrinsics
                    fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
                    cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
                    
                    x_3d = (x - cx) * depth / fx
                    y_3d = (y - cy) * depth / fy
                    z_3d = depth
                    
                    points_3d.append([x_3d, y_3d, z_3d])
                else:
                    points_3d.append([0, 0, 0])
            else:
                points_3d.append([0, 0, 0])
                
        return np.array(points_3d)
        
    def estimate_transform_ransac(self, src_points, dst_points, max_iterations=100):
        """Estimate transformation using RANSAC"""
        if len(src_points) < 3:
            return None
            
        best_inliers = 0
        best_transform = None
        threshold = 0.05  # 5cm threshold
        
        for _ in range(max_iterations):
            # Select random points
            indices = np.random.choice(len(src_points), min(3, len(src_points)), replace=False)
            src_sample = src_points[indices]
            dst_sample = dst_points[indices]
            
            # Calculate transformation (simple translation)
            transform = np.mean(dst_sample - src_sample, axis=0)
            
            # Count inliers
            transformed_src = src_points + transform
            distances = np.linalg.norm(transformed_src - dst_points, axis=1)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_transform = transform
                
        # If we have enough inliers, refine the transformation
        if best_inliers >= min(5, len(src_points) // 2):
            # Refine using all inliers
            transformed_src = src_points + best_transform
            distances = np.linalg.norm(transformed_src - dst_points, axis=1)
            inlier_mask = distances < threshold
            
            if np.sum(inlier_mask) > 0:
                refined_transform = np.mean(dst_points[inlier_mask] - src_points[inlier_mask], axis=0)
                
                # Limit movement to reasonable values
                if np.linalg.norm(refined_transform) < 1.0:  # Max 1 meter per frame
                    return refined_transform
                    
        return None
        
    def get_current_features(self):
        """Get current feature points for visualization"""
        return self.current_features
        
    def reset(self):
        """Reset the visual odometry"""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.current_features = []
        self.movement_history.clear()
        print("🔄 Visual odometry reset")