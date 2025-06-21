# enhanced_odometry.py
#
# Contains the implementation of an enhanced Visual Odometry system that uses
# Keyframes to reduce drift and improve tracking accuracy. This is the first
# major step in upgrading the system's core tracking logic.

import cv2
import numpy as np

# --- Helper Class: KeyFrame ---
# A KeyFrame stores important information from a past moment in time.
# Instead of tracking from one frame to the next, we track from a stable
# KeyFrame, which acts as a more reliable anchor point.
class KeyFrame:
    """
    Represents a KeyFrame in our Visual Odometry system.
    It holds the data needed to act as a stable reference for tracking.
    """
    def __init__(self, frame, depth_map, keypoints, descriptors, camera_intrinsics):
        self.image = frame
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.pose = np.eye(4)  # Represents position and orientation in the world

        # Convert 2D feature points to 3D world points using the depth map
        self.points_3d = self._features_to_3d(depth_map, camera_intrinsics)

    def _features_to_3d(self, depth_map, K):
        """
        Converts keypoints to 3D points using the depth map and camera intrinsics.
        K is the camera intrinsic matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        """
        points_3d = []
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        for kp in self.keypoints:
            u, v = int(kp.pt[0]), int(kp.pt[1])

            # Ensure the point is within the bounds of the depth map
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                # Get depth value. RealSense depth is often in millimeters (uint16).
                # Convert to meters.
                z = depth_map[v, u] * 0.001  # Assuming depth scale of 0.001

                # Only use points with a valid depth reading
                if z > 0:
                    # De-projection formula
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points_3d.append([x, y, z])
            
        return np.array(points_3d, dtype=np.float32)

# --- Main Class: Enhanced Odometry ---
class EnhancedOdometry:
    """
    Manages the Visual Odometry process using a KeyFrame-based approach
    to improve accuracy and reduce long-term drift.
    """
    def __init__(self, camera_intrinsics):
        # Store camera parameters
        self.K = camera_intrinsics
        
        # Initialize the feature detector and matcher
        # ORB is a great choice as it's fast and effective.
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # State variables
        self.keyframes = []
        self.current_pose = np.eye(4) # Start at the origin
        
        print("Enhanced Odometry Initialized.")

    def _detect_and_compute(self, frame):
        """Detects features and computes descriptors for a given frame."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.orb.detectAndCompute(gray_frame, None)
        return keypoints, descriptors

    def process_frame(self, frame, depth_map):
        """
        Main function to process a new frame and update the camera pose.
        This implements the core logic of tracking against a KeyFrame.
        """
        keypoints, descriptors = self._detect_and_compute(frame)

        if not self.keyframes:
            # --- First Frame: Create the initial KeyFrame ---
            print("First frame: Creating initial KeyFrame.")
            new_keyframe = KeyFrame(frame, depth_map, keypoints, descriptors, self.K)
            new_keyframe.pose = self.current_pose
            self.keyframes.append(new_keyframe)
            return self.current_pose

        # --- Subsequent Frames: Track against the latest KeyFrame ---
        last_keyframe = self.keyframes[-1]
        
        # Match features between the current frame and the last keyframe
        matches = self.matcher.match(descriptors, last_keyframe.descriptors)
        matches = sorted(matches, key=lambda x: x.distance) # Sort by quality
        
        # We need a minimum number of matches to attempt tracking
        if len(matches) < 30:
            print("Tracking failed: Not enough matches.")
            # Here you might add logic to handle tracking loss (e.g., try matching against older keyframes)
            return self.current_pose

        # Get the 3D points from the keyframe and 2D points from the current frame for our matches
        points_3d_object = np.array([last_keyframe.points_3d[m.trainIdx] for m in matches])
        points_2d_image = np.array([keypoints[m.queryIdx].pt for m in matches])

        # --- Use solvePnP to estimate the camera's pose relative to the KeyFrame ---
        # solvePnPRansac is robust against outlier matches.
        try:
            _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d_object, points_2d_image, self.K, None)
            
            # Convert rotation vector to a rotation matrix
            R, _ = cv2.Rodrigues(rvec)

            # --- Update the current pose ---
            # The transformation from the keyframe to the current camera
            relative_transform = np.eye(4)
            relative_transform[0:3, 0:3] = R
            relative_transform[0:3, 3] = tvec.flatten()
            
            # To get the current world pose, we multiply the keyframe's world pose
            # by the relative transform we just found.
            self.current_pose = np.dot(last_keyframe.pose, np.linalg.inv(relative_transform))

        except cv2.error as e:
            print(f"Error during pose estimation: {e}")
            return self.current_pose
        
        # --- Simple KeyFrame creation logic (to be enhanced later) ---
        # If the number of inliers is low, it means we've moved significantly
        # and should create a new KeyFrame to act as our new anchor.
        if inliers is not None and len(inliers) < 50:
            print(f"New KeyFrame created. Inliers: {len(inliers)}")
            new_keyframe = KeyFrame(frame, depth_map, keypoints, descriptors, self.K)
            new_keyframe.pose = self.current_pose # Its pose is the one we just calculated
            self.keyframes.append(new_keyframe)

        return self.current_pose