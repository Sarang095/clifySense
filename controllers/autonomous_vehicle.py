"""
Autonomous Vehicle Controller for Webots Simulation.

This controller serves as the interface between Webots and the AI models,
handling sensor data collection, action execution, and environment interaction.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import yaml
import time

# Add project root to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from controller import Robot, Camera, DistanceSensor, GPS, Compass, Motor
except ImportError:
    print("Warning: Webots controller library not available. Running in simulation mode.")
    # Mock classes for development/testing outside Webots
    class Robot:
        def __init__(self): pass
        def getTimeStep(self): return 32
        def step(self, time_step): return 0
        def getDevice(self, name): return MockDevice()
    
    class MockDevice:
        def enable(self, time_step): pass
        def getValues(self): return [1.0] * 8
        def getValue(self): return 0.0
        def getImage(self): return np.zeros((224, 224, 3), dtype=np.uint8)
        def setPosition(self, pos): pass
        def setVelocity(self, vel): pass
        def getWidth(self): return 224
        def getHeight(self): return 224

from utils.sensor_processing import StateExtractor


class AutonomousVehicle:
    """Main autonomous vehicle controller for Webots simulation."""
    
    def __init__(self, config_path: str = "config/environment_config.yaml"):
        """Initialize the autonomous vehicle controller."""
        self.robot = Robot()
        self.time_step = int(self.robot.getTimeStep())
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize sensors and actuators
        self._initialize_devices()
        
        # Initialize state processor
        self.state_extractor = StateExtractor(config_path)
        
        # Vehicle state
        self.current_state = None
        self.episode_step = 0
        self.total_reward = 0.0
        
        # Control limits
        self.max_speed = self.config['vehicle']['max_speed'] / 3.6  # Convert km/h to m/s
        self.max_steering = self.config['vehicle']['max_steering_angle']
        
        print(f"Autonomous vehicle initialized with time step: {self.time_step}ms")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if config file is not available."""
        return {
            'vehicle': {
                'max_speed': 30.0,
                'max_steering_angle': 0.6,
                'camera_resolution': [224, 224],
                'sensor_update_rate': 20
            },
            'environments': {
                'lane': {
                    'rewards': {
                        'lane_keeping': 1.0,
                        'collision': -100.0,
                        'goal_reached': 50.0,
                        'speed_penalty': -0.1
                    }
                }
            }
        }
    
    def _initialize_devices(self):
        """Initialize all vehicle sensors and actuators."""
        # Camera
        try:
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(self.time_step)
            print("Camera initialized")
        except:
            print("Warning: Camera not found")
            self.camera = None
        
        # Distance sensors (assuming 8 sensors around the vehicle)
        self.distance_sensors = []
        sensor_names = ["ds_front", "ds_front_left", "ds_front_right", "ds_left", 
                       "ds_right", "ds_rear_left", "ds_rear_right", "ds_rear"]
        
        for name in sensor_names:
            try:
                sensor = self.robot.getDevice(name)
                sensor.enable(self.time_step)
                self.distance_sensors.append(sensor)
            except:
                # If specific sensors don't exist, try generic names
                try:
                    sensor = self.robot.getDevice(f"distance_sensor_{len(self.distance_sensors)}")
                    sensor.enable(self.time_step)
                    self.distance_sensors.append(sensor)
                except:
                    pass
        
        print(f"Initialized {len(self.distance_sensors)} distance sensors")
        
        # GPS for position tracking
        try:
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.time_step)
            print("GPS initialized")
        except:
            print("Warning: GPS not found")
            self.gps = None
        
        # Compass for orientation
        try:
            self.compass = self.robot.getDevice("compass")
            self.compass.enable(self.time_step)
            print("Compass initialized")
        except:
            print("Warning: Compass not found")
            self.compass = None
        
        # Motors (wheels)
        try:
            self.left_motor = self.robot.getDevice("left_wheel_motor")
            self.right_motor = self.robot.getDevice("right_wheel_motor")
            
            # Set motors to velocity control mode
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            print("Motors initialized")
        except:
            print("Warning: Motors not found")
            self.left_motor = None
            self.right_motor = None
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """Collect data from all sensors."""
        sensor_data = {}
        
        # Camera image
        if self.camera:
            try:
                image = self.camera.getImage()
                width = self.camera.getWidth()
                height = self.camera.getHeight()
                
                # Convert Webots image to numpy array
                image_array = np.frombuffer(image, dtype=np.uint8)
                image_array = image_array.reshape((height, width, 4))  # BGRA format
                sensor_data['camera_image'] = image_array[:, :, :3]  # Remove alpha channel
            except:
                sensor_data['camera_image'] = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            sensor_data['camera_image'] = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Distance sensors
        distance_values = []
        for sensor in self.distance_sensors:
            try:
                distance_values.append(sensor.getValue())
            except:
                distance_values.append(10.0)  # Max range if sensor fails
        
        # Pad to ensure we have at least 8 values
        while len(distance_values) < 8:
            distance_values.append(10.0)
        
        sensor_data['distance_sensors'] = distance_values
        
        # GPS position
        if self.gps:
            try:
                position = self.gps.getValues()
                sensor_data['position'] = position
            except:
                sensor_data['position'] = [0.0, 0.0, 0.0]
        else:
            sensor_data['position'] = [0.0, 0.0, 0.0]
        
        # Compass heading
        if self.compass:
            try:
                north = self.compass.getValues()
                heading = np.arctan2(north[0], north[2])
                sensor_data['heading'] = heading
            except:
                sensor_data['heading'] = 0.0
        else:
            sensor_data['heading'] = 0.0
        
        return sensor_data
    
    def execute_action(self, action: np.ndarray):
        """
        Execute vehicle action.
        
        Args:
            action: Array containing [throttle, steering] values in range [-1, 1]
        """
        if len(action) != 2:
            print(f"Warning: Expected action of length 2, got {len(action)}")
            return
        
        throttle, steering = action
        
        # Clamp actions to valid range
        throttle = np.clip(throttle, -1.0, 1.0)
        steering = np.clip(steering, -1.0, 1.0)
        
        # Convert to motor velocities
        max_motor_velocity = self.max_speed / 0.1  # Assuming wheel radius of 0.1m
        base_velocity = throttle * max_motor_velocity
        
        # Apply differential steering
        steering_factor = steering * 0.5  # Adjust steering sensitivity
        left_velocity = base_velocity - steering_factor * max_motor_velocity
        right_velocity = base_velocity + steering_factor * max_motor_velocity
        
        # Set motor velocities
        if self.left_motor and self.right_motor:
            try:
                self.left_motor.setVelocity(left_velocity)
                self.right_motor.setVelocity(right_velocity)
            except:
                print("Warning: Failed to set motor velocities")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current processed state of the vehicle."""
        sensor_data = self.get_sensor_data()
        
        # Calculate current speed (simplified)
        current_speed = 0.0  # Would need wheel encoders for accurate measurement
        current_steering = 0.0  # Would need steering angle sensor
        
        # Extract state using state processor
        position_3d = (sensor_data['position'][0], sensor_data['position'][2], sensor_data['heading'])
        
        state = self.state_extractor.extract_state(
            camera_image=sensor_data['camera_image'],
            distance_sensors=sensor_data['distance_sensors'],
            speed=current_speed,
            steering_angle=current_steering,
            position=position_3d
        )
        
        self.current_state = state
        return state
    
    def calculate_reward(self, state: Dict[str, Any], action: np.ndarray, 
                        environment: str = "lane") -> float:
        """
        Calculate reward based on current state and action.
        
        Args:
            state: Current state dictionary
            action: Last action taken
            environment: Current environment type
            
        Returns:
            Calculated reward value
        """
        if environment not in self.config['environments']:
            environment = "lane"  # Default to lane following
        
        rewards_config = self.config['environments'][environment]['rewards']
        reward = 0.0
        
        # Lane keeping reward (for lane following)
        if 'lane_keeping' in rewards_config:
            lane_deviation = abs(state.get('lane_deviation', 0.0))
            lane_reward = rewards_config['lane_keeping'] * (1.0 - lane_deviation)
            reward += lane_reward
        
        # Collision penalty
        if 'collision' in rewards_config:
            obstacles = state.get('obstacles', {})
            if not obstacles.get('is_safe', True):
                reward += rewards_config['collision']
        
        # Speed penalty (encourage appropriate speed)
        if 'speed_penalty' in rewards_config:
            speed = state.get('speed', 0.0)
            if speed < 0.3:  # Too slow
                reward += rewards_config['speed_penalty']
        
        return reward
    
    def reset(self):
        """Reset the vehicle to initial state."""
        # Stop the vehicle
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        
        # Reset counters
        self.episode_step = 0
        self.total_reward = 0.0
        
        # Allow time for physics to settle
        for _ in range(10):
            self.robot.step(self.time_step)
        
        # Get initial state
        return self.get_current_state()
    
    def step(self, action: np.ndarray, environment: str = "lane") -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Execute one simulation step.
        
        Args:
            action: Action to execute [throttle, steering]
            environment: Current environment type
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute action
        self.execute_action(action)
        
        # Step the simulation
        if self.robot.step(self.time_step) == -1:
            # Simulation ended
            return self.current_state, 0.0, True, {'reason': 'simulation_ended'}
        
        # Get new state
        next_state = self.get_current_state()
        
        # Calculate reward
        reward = self.calculate_reward(next_state, action, environment)
        self.total_reward += reward
        
        # Check termination conditions
        done = False
        info = {}
        
        # Episode length limit
        self.episode_step += 1
        max_steps = self.config['environments'].get(environment, {}).get('max_episode_steps', 1000)
        if self.episode_step >= max_steps:
            done = True
            info['reason'] = 'max_steps_reached'
        
        # Collision check
        obstacles = next_state.get('obstacles', {})
        if not obstacles.get('is_safe', True):
            done = True
            info['reason'] = 'collision'
        
        info.update({
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'lane_deviation': next_state.get('lane_deviation', 0.0)
        })
        
        return next_state, reward, done, info
    
    def run_demo(self, duration: float = 60.0):
        """
        Run a simple demo with basic lane following behavior.
        
        Args:
            duration: Demo duration in seconds
        """
        print(f"Starting demo for {duration} seconds...")
        start_time = time.time()
        
        # Reset vehicle
        self.reset()
        
        while time.time() - start_time < duration:
            # Get current state
            state = self.get_current_state()
            
            # Simple lane following behavior
            lane_deviation = state.get('lane_deviation', 0.0)
            obstacles = state.get('obstacles', {})
            
            # Calculate steering to center in lane
            steering = -lane_deviation * 2.0  # Proportional control
            
            # Calculate throttle based on obstacles
            if obstacles.get('front', 1.0) < 0.5:
                throttle = -0.3  # Brake if obstacle ahead
            else:
                throttle = 0.3  # Maintain moderate speed
            
            # Execute action
            action = np.array([throttle, steering])
            state, reward, done, info = self.step(action)
            
            # Print status
            if self.episode_step % 50 == 0:
                print(f"Step: {self.episode_step}, "
                      f"Lane deviation: {lane_deviation:.3f}, "
                      f"Reward: {reward:.2f}, "
                      f"Safe: {obstacles.get('is_safe', True)}")
            
            if done:
                print(f"Episode ended: {info.get('reason', 'unknown')}")
                break
        
        print("Demo completed!")


def main():
    """Main function to run the autonomous vehicle controller."""
    # Initialize vehicle
    vehicle = AutonomousVehicle()
    
    # Run demo
    vehicle.run_demo(duration=120.0)  # 2 minute demo


if __name__ == "__main__":
    main()