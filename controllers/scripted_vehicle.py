"""
Scripted Vehicle Controller for Dynamic Obstacles.

This controller creates simple scripted behavior for other vehicles
that act as dynamic obstacles in the simulation.
"""

import sys
import os
import numpy as np
import time

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from controller import Robot, Motor, GPS, Compass
except ImportError:
    print("Warning: Webots controller library not available.")
    
    class Robot:
        def __init__(self): pass
        def getTimeStep(self): return 32
        def step(self, time_step): return 0
        def getDevice(self, name): return MockDevice()
    
    class MockDevice:
        def setPosition(self, pos): pass
        def setVelocity(self, vel): pass
        def enable(self, time_step): pass
        def getValues(self): return [0.0, 0.0, 0.0]


class ScriptedVehicle:
    """Scripted vehicle that follows predefined patterns."""
    
    def __init__(self, behavior_type="circular"):
        """
        Initialize scripted vehicle.
        
        Args:
            behavior_type: Type of behavior ("circular", "linear", "stop_and_go")
        """
        self.robot = Robot()
        self.time_step = int(self.robot.getTimeStep())
        self.behavior_type = behavior_type
        
        # Initialize motors
        self._initialize_motors()
        
        # Initialize sensors
        self._initialize_sensors()
        
        # Behavior parameters
        self.start_time = time.time()
        self.behavior_state = 0
        self.max_speed = 5.0  # m/s
        
        print(f"Scripted vehicle initialized with {behavior_type} behavior")
    
    def _initialize_motors(self):
        """Initialize vehicle motors."""
        try:
            self.left_motor = self.robot.getDevice("left_wheel_motor")
            self.right_motor = self.robot.getDevice("right_wheel_motor")
            
            # Set motors to velocity control mode
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except:
            self.left_motor = None
            self.right_motor = None
            print("Warning: Motors not found in scripted vehicle")
    
    def _initialize_sensors(self):
        """Initialize sensors for position tracking."""
        try:
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.time_step)
        except:
            self.gps = None
        
        try:
            self.compass = self.robot.getDevice("compass")
            self.compass.enable(self.time_step)
        except:
            self.compass = None
    
    def get_position(self):
        """Get current vehicle position."""
        if self.gps:
            try:
                return self.gps.getValues()
            except:
                return [0.0, 0.0, 0.0]
        return [0.0, 0.0, 0.0]
    
    def get_heading(self):
        """Get current vehicle heading."""
        if self.compass:
            try:
                north = self.compass.getValues()
                return np.arctan2(north[0], north[2])
            except:
                return 0.0
        return 0.0
    
    def set_velocity(self, linear_velocity, angular_velocity):
        """
        Set vehicle velocity.
        
        Args:
            linear_velocity: Forward velocity (m/s)
            angular_velocity: Angular velocity (rad/s)
        """
        if not self.left_motor or not self.right_motor:
            return
        
        # Convert to wheel velocities (differential drive)
        wheel_radius = 0.3  # Approximate wheel radius
        wheel_separation = 1.6  # Approximate distance between wheels
        
        # Calculate wheel velocities
        left_vel = (linear_velocity - angular_velocity * wheel_separation / 2) / wheel_radius
        right_vel = (linear_velocity + angular_velocity * wheel_separation / 2) / wheel_radius
        
        try:
            self.left_motor.setVelocity(left_vel)
            self.right_motor.setVelocity(right_vel)
        except:
            pass
    
    def circular_behavior(self):
        """Circular driving pattern."""
        elapsed_time = time.time() - self.start_time
        
        # Drive in a circle
        linear_vel = 2.0  # m/s
        angular_vel = 0.3  # rad/s
        
        # Vary speed slightly for realism
        speed_variation = 0.5 * np.sin(elapsed_time * 0.5)
        linear_vel += speed_variation
        
        self.set_velocity(linear_vel, angular_vel)
    
    def linear_behavior(self):
        """Linear driving pattern with occasional turns."""
        elapsed_time = time.time() - self.start_time
        
        # Change behavior every 10 seconds
        cycle_time = elapsed_time % 20
        
        if cycle_time < 8:
            # Drive straight
            self.set_velocity(3.0, 0.0)
        elif cycle_time < 12:
            # Turn left
            self.set_velocity(1.0, 0.5)
        elif cycle_time < 16:
            # Drive straight
            self.set_velocity(3.0, 0.0)
        else:
            # Turn right
            self.set_velocity(1.0, -0.5)
    
    def stop_and_go_behavior(self):
        """Stop and go traffic pattern."""
        elapsed_time = time.time() - self.start_time
        
        # Cycle: 5s driving, 3s stopped
        cycle_time = elapsed_time % 8
        
        if cycle_time < 5:
            # Drive
            self.set_velocity(2.5, 0.0)
        else:
            # Stop
            self.set_velocity(0.0, 0.0)
    
    def lane_change_behavior(self):
        """Lane changing behavior."""
        elapsed_time = time.time() - self.start_time
        
        # Lane change every 15 seconds
        cycle_time = elapsed_time % 15
        
        if cycle_time < 3:
            # Turn slightly left (lane change)
            self.set_velocity(2.0, 0.2)
        elif cycle_time < 6:
            # Straighten out
            self.set_velocity(2.5, -0.1)
        elif cycle_time < 9:
            # Normal driving
            self.set_velocity(3.0, 0.0)
        elif cycle_time < 12:
            # Turn slightly right (return to original lane)
            self.set_velocity(2.0, -0.2)
        else:
            # Straighten out
            self.set_velocity(2.5, 0.1)
    
    def run(self):
        """Main control loop."""
        print(f"Starting scripted vehicle with {self.behavior_type} behavior")
        
        while self.robot.step(self.time_step) != -1:
            # Execute behavior based on type
            if self.behavior_type == "circular":
                self.circular_behavior()
            elif self.behavior_type == "linear":
                self.linear_behavior()
            elif self.behavior_type == "stop_and_go":
                self.stop_and_go_behavior()
            elif self.behavior_type == "lane_change":
                self.lane_change_behavior()
            else:
                # Default: slow circular motion
                self.set_velocity(1.0, 0.1)


def main():
    """Main function for scripted vehicle controller."""
    # Determine behavior based on robot name or random selection
    import random
    
    behaviors = ["circular", "linear", "stop_and_go", "lane_change"]
    behavior = random.choice(behaviors)
    
    vehicle = ScriptedVehicle(behavior_type=behavior)
    vehicle.run()


if __name__ == "__main__":
    main()