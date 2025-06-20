"""
Agricultural Image Filters
Specialized image processing for agricultural environments
Handles lighting variations, dust, vibration, and seasonal changes
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional
import time

class AgriculturalImageProcessor:
    """
    Image processing pipeline optimized for agricultural environments
    Handles outdoor lighting, dust, vibration, and seasonal variations
    """
    
    def __init__(self):
        """Initialize agricultural image processor"""
        # CLAHE for lighting normalization
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Bilateral filter for noise reduction
        self.bilateral_d = 9
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75
        
        # Gaussian blur for vibration compensation
        self.vibration_kernel_size = (3, 3)
        self.vibration_sigma = 0.8
        
        # Adaptive parameters
        self.lighting_adaptation = True
        self.dust_filtering = True
        self.vibration_compensation = True
        
        print("Agricultural Image Processor initialized")
    
    def process_agricultural_frame(self, frame: np.ndarray, 
                                 depth_frame: np.ndarray = None) -> Dict:
        """
        Complete agricultural frame processing pipeline
        
        Args:
            frame: Input color frame
            depth_frame: Optional depth frame
            
        Returns:
            Dictionary with processed frames and metadata
        """
        try:
            start_time = time.time()
            
            # Convert to grayscale for processing
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # 1. Lighting normalization
            enhanced_gray = self.normalize_lighting(gray)
            
            # 2. Noise reduction
            denoised = self.reduce_agricultural_noise(enhanced_gray)
            
            # 3. Vibration compensation
            stabilized = self.compensate_vibration(denoised)
            
            # 4. Dust and particle filtering
            clean = self.filter_dust_particles(stabilized)
            
            # Convert back to color if needed
            if len(frame.shape) == 3:
                processed_color = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
                # Enhance color channels
                processed_color = self.enhance_color_channels(frame, processed_color)
            else:
                processed_color = clean
            
            # Process depth frame if available
            processed_depth = None
            if depth_frame is not None:
                processed_depth = self.process_depth_agricultural(depth_frame)
            
            processing_time = time.time() - start_time
            
            return {
                'processed_frame': processed_color,
                'processed_depth': processed_depth,
                'enhanced_gray': clean,
                'processing_time': processing_time,
                'metadata': {
                    'lighting_normalized': True,
                    'noise_reduced': True,
                    'vibration_compensated': True,
                    'dust_filtered': True
                }
            }
            
        except Exception as e:
            print(f"Agricultural frame processing error: {e}")
            return {
                'processed_frame': frame,
                'processed_depth': depth_frame,
                'enhanced_gray': frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                'processing_time': 0.0,
                'metadata': {'error': str(e)}
            }
    
    def normalize_lighting(self, gray_frame: np.ndarray) -> np.ndarray:
        """Normalize lighting for outdoor agricultural conditions"""
        try:
            if not self.lighting_adaptation:
                return gray_frame
            
            # Apply CLAHE for adaptive histogram equalization
            enhanced = self.clahe.apply(gray_frame)
            
            # Additional gamma correction for very dark/bright conditions
            mean_intensity = np.mean(enhanced)
            
            if mean_intensity < 80:  # Very dark
                gamma = 0.7
                enhanced = np.power(enhanced / 255.0, gamma) * 255.0
                enhanced = enhanced.astype(np.uint8)
            elif mean_intensity > 180:  # Very bright
                gamma = 1.3
                enhanced = np.power(enhanced / 255.0, gamma) * 255.0
                enhanced = enhanced.astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            print(f"Lighting normalization error: {e}")
            return gray_frame
    
    def reduce_agricultural_noise(self, frame: np.ndarray) -> np.ndarray:
        """Reduce noise common in agricultural environments"""
        try:
            if not self.dust_filtering:
                return frame
            
            # Bilateral filter for edge-preserving noise reduction
            denoised = cv2.bilateralFilter(
                frame, 
                self.bilateral_d, 
                self.bilateral_sigma_color, 
                self.bilateral_sigma_space
            )
            
            return denoised
            
        except Exception as e:
            print(f"Noise reduction error: {e}")
            return frame
    
    def compensate_vibration(self, frame: np.ndarray) -> np.ndarray:
        """Compensate for vibration from agricultural machinery"""
        try:
            if not self.vibration_compensation:
                return frame
            
            # Gentle Gaussian blur to reduce high-frequency vibration artifacts
            stabilized = cv2.GaussianBlur(frame, self.vibration_kernel_size, self.vibration_sigma)
            
            return stabilized
            
        except Exception as e:
            print(f"Vibration compensation error: {e}")
            return frame
    
    def filter_dust_particles(self, frame: np.ndarray) -> np.ndarray:
        """Filter dust particles and small debris"""
        try:
            # Morphological opening to remove small particles
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            
            # Median filter for additional particle removal
            filtered = cv2.medianBlur(opened, 3)
            
            return filtered
            
        except Exception as e:
            print(f"Dust filtering error: {e}")
            return frame
    
    def enhance_color_channels(self, original: np.ndarray, processed_gray: np.ndarray) -> np.ndarray:
        """Enhance color channels while preserving processed grayscale improvements"""
        try:
            if len(original.shape) != 3 or len(processed_gray.shape) != 3:
                return processed_gray
            
            # Convert to LAB color space
            lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
            
            # Replace L channel with processed grayscale
            lab[:, :, 0] = cv2.cvtColor(processed_gray, cv2.COLOR_BGR2GRAY)
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return enhanced
            
        except Exception as e:
            print(f"Color enhancement error: {e}")
            return processed_gray
    
    def process_depth_agricultural(self, depth_frame: np.ndarray) -> np.ndarray:
        """Process depth frame for agricultural environments"""
        try:
            # Remove depth noise and holes
            processed = depth_frame.copy()
            
            # Fill small holes in depth data
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = (processed == 0).astype(np.uint8)
            processed = cv2.inpaint(processed.astype(np.uint16), mask, 3, cv2.INPAINT_TELEA)
            
            # Median filter for depth noise
            processed = cv2.medianBlur(processed, 5)
            
            return processed
            
        except Exception as e:
            print(f"Depth processing error: {e}")
            return depth_frame
    
    def adapt_to_lighting_conditions(self, frame: np.ndarray):
        """Adapt processing parameters based on current lighting"""
        try:
            mean_intensity = np.mean(frame)
            
            # Adjust CLAHE parameters based on lighting
            if mean_intensity < 60:  # Low light
                self.clahe.setClipLimit(4.0)
                self.clahe.setTilesGridSize((6, 6))
            elif mean_intensity > 200:  # Bright light
                self.clahe.setClipLimit(2.0)
                self.clahe.setTilesGridSize((10, 10))
            else:  # Normal light
                self.clahe.setClipLimit(3.0)
                self.clahe.setTilesGridSize((8, 8))
                
        except Exception as e:
            print(f"Lighting adaptation error: {e}")

# Global instance
agricultural_processor = AgriculturalImageProcessor()

def process_agricultural_frame(frame: np.ndarray, depth_frame: np.ndarray = None) -> Dict:
    """Global function for processing agricultural frames"""
    return agricultural_processor.process_agricultural_frame(frame, depth_frame)