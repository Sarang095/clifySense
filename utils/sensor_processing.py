"""Sensor data processing utilities for autonomous driving."""

import numpy as np
import cv2
from typing import Tuple, Optional, List
import yaml


class CameraProcessor:
    """Processes camera input for the autonomous driving agent."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess camera image for neural network input.
        
        Args:
            image: Raw camera image from Webots
            
        Returns:
            Preprocessed image normalized and resized
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            # Remove alpha channel if present
            image = image[:, :, :3]
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def extract_lane_features(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract lane line features from camera image.
        
        Args:
            image: Preprocessed camera image
            
        Returns:
            Tuple of (lane_mask, lane_points)
        """
        # Convert to HSV for better lane detection
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        
        # Define range for white color (lane markings)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 25, 255])
        
        # Create mask for lane markings
        lane_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Find contours
        contours, _ = cv2.findContours(lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract lane points
        lane_points = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    lane_points.append([cx, cy])
        
        return lane_mask, np.array(lane_points)


class DistanceSensorProcessor:
    """Processes distance sensor data for obstacle detection."""
    
    def __init__(self, max_range: float = 10.0):
        self.max_range = max_range
        
    def process_distance_sensors(self, sensor_values: List[float]) -> dict:
        """
        Process distance sensor readings to detect obstacles.
        
        Args:
            sensor_values: List of distance sensor readings
            
        Returns:
            Dictionary containing obstacle information
        """
        # Normalize sensor values
        normalized_values = [min(val, self.max_range) / self.max_range for val in sensor_values]
        
        # Detect obstacles in different zones
        obstacles = {
            'front': min(normalized_values[0:3]) if len(normalized_values) >= 3 else 1.0,
            'left': min(normalized_values[3:6]) if len(normalized_values) >= 6 else 1.0,
            'right': min(normalized_values[6:9]) if len(normalized_values) >= 9 else 1.0,
            'rear': min(normalized_values[9:12]) if len(normalized_values) >= 12 else 1.0,
        }
        
        # Calculate safe distances
        obstacles['is_safe'] = all(dist > 0.3 for dist in obstacles.values())
        obstacles['closest_obstacle'] = min(obstacles.values())
        
        return obstacles


class StateExtractor:
    """Extracts state representation from sensor data."""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
            
        self.camera_processor = CameraProcessor(
            tuple(self.config['vehicle']['camera_resolution'])
        )
        self.distance_processor = DistanceSensorProcessor()
    
    def _default_config(self) -> dict:
        """Default configuration if no config file provided."""
        return {
            'vehicle': {
                'camera_resolution': [224, 224],
                'max_speed': 30.0,
                'max_steering_angle': 0.6
            }
        }
    
    def extract_state(self, camera_image: np.ndarray, 
                     distance_sensors: List[float],
                     speed: float,
                     steering_angle: float,
                     position: Tuple[float, float, float]) -> dict:
        """
        Extract complete state representation.
        
        Args:
            camera_image: Raw camera image
            distance_sensors: Distance sensor readings
            speed: Current vehicle speed
            steering_angle: Current steering angle
            position: Vehicle position (x, y, heading)
            
        Returns:
            Dictionary containing processed state information
        """
        # Process camera image
        processed_image = self.camera_processor.preprocess_image(camera_image)
        lane_mask, lane_points = self.camera_processor.extract_lane_features(processed_image)
        
        # Process distance sensors
        obstacles = self.distance_processor.process_distance_sensors(distance_sensors)
        
        # Normalize vehicle state
        normalized_speed = speed / self.config['vehicle']['max_speed']
        normalized_steering = steering_angle / self.config['vehicle']['max_steering_angle']
        
        # Calculate lane deviation
        image_center = processed_image.shape[1] // 2
        if len(lane_points) > 0:
            avg_lane_center = np.mean(lane_points[:, 0])
            lane_deviation = (avg_lane_center - image_center) / image_center
        else:
            lane_deviation = 0.0
        
        state = {
            'image': processed_image,
            'lane_mask': lane_mask,
            'lane_points': lane_points,
            'lane_deviation': lane_deviation,
            'obstacles': obstacles,
            'speed': normalized_speed,
            'steering_angle': normalized_steering,
            'position': position,
            'vector_state': np.array([
                normalized_speed,
                normalized_steering,
                lane_deviation,
                obstacles['front'],
                obstacles['left'],
                obstacles['right'],
                obstacles['rear']
            ])
        }
        
        return state