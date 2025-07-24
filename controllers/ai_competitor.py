#!/usr/bin/env python3
"""
AI Competitor Controller for Webots Driving Game
This AI car follows the racing circuit and provides competition for the player.
"""

import sys
import math
import time
from controller import Robot, GPS, Compass, DistanceSensor

class AICompetitorController:
    def __init__(self):
        # Initialize the robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # AI state
        self.current_waypoint = 0
        self.speed = 40  # Base speed
        self.max_speed = 50
        self.min_speed = 20
        
        # Racing waypoints (following the track)
        self.waypoints = [
            (0, -75),    # Start/Finish
            (20, -60),   # Turn towards checkpoint 2
            (50, -30),   # Approaching turn
            (75, 0),     # Checkpoint 2
            (60, 20),    # Turn towards checkpoint 3
            (30, 50),    # Approaching turn
            (0, 75),     # Checkpoint 3
            (-20, 60),   # Turn towards checkpoint 4
            (-50, 30),   # Approaching turn
            (-75, 0),    # Checkpoint 4
            (-60, -20),  # Turn towards start
            (-30, -50),  # Approaching turn
        ]
        
        # Initialize car components
        self.init_car_components()
        self.init_sensors()
        
        print("🤖 AI Competitor initialized!")

    def init_car_components(self):
        """Initialize car motors and basic components."""
        # Get the four wheels
        self.wheels = []
        wheel_names = ["front left wheel", "front right wheel", 
                      "rear left wheel", "rear right wheel"]
        
        for name in wheel_names:
            wheel = self.robot.getDevice(name)
            if wheel:
                wheel.setPosition(float('inf'))
                wheel.setVelocity(0.0)
                self.wheels.append(wheel)
        
        # Steering motors
        self.steering = []
        steering_names = ["front left wheel steering motor", "front right wheel steering motor"]
        
        for name in steering_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(0.0)
                self.steering.append(motor)

    def init_sensors(self):
        """Initialize sensors."""
        # GPS for position tracking
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.timestep)
        
        # Compass for orientation
        self.compass = self.robot.getDevice("compass")
        if self.compass:
            self.compass.enable(self.timestep)

    def get_position(self):
        """Get current car position."""
        if self.gps:
            return self.gps.getValues()
        return [0, 0, 0]

    def get_orientation(self):
        """Get current car orientation angle."""
        if self.compass:
            compass_values = self.compass.getValues()
            return math.atan2(compass_values[0], compass_values[2])
        return 0

    def calculate_steering_angle(self):
        """Calculate steering angle to reach current waypoint."""
        pos = self.get_position()
        current_x, current_z = pos[0], pos[2]
        
        # Get target waypoint
        target_x, target_z = self.waypoints[self.current_waypoint]
        
        # Calculate distance to waypoint
        distance = math.sqrt((target_x - current_x)**2 + (target_z - current_z)**2)
        
        # If close to waypoint, move to next one
        if distance < 10:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
            target_x, target_z = self.waypoints[self.current_waypoint]
        
        # Calculate angle to target
        target_angle = math.atan2(target_x - current_x, target_z - current_z)
        current_angle = self.get_orientation()
        
        # Calculate steering angle
        angle_diff = target_angle - current_angle
        
        # Normalize angle to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Convert to steering angle (limit to reasonable range)
        steering_angle = angle_diff * 0.5
        steering_angle = max(-0.6, min(0.6, steering_angle))
        
        return steering_angle

    def calculate_speed(self, steering_angle):
        """Calculate appropriate speed based on steering angle."""
        # Slow down for sharp turns
        turn_factor = abs(steering_angle) / 0.6  # Normalize to 0-1
        speed_reduction = turn_factor * 0.4  # Reduce speed by up to 40%
        
        adjusted_speed = self.speed * (1 - speed_reduction)
        return max(self.min_speed, min(self.max_speed, adjusted_speed))

    def avoid_obstacles(self, base_steering, base_speed):
        """Simple obstacle avoidance (if sensors are available)."""
        # This is a basic implementation
        # In a real scenario, you'd use distance sensors
        return base_steering, base_speed

    def set_car_movement(self, speed, steering_angle):
        """Set car movement based on speed and steering angle."""
        # Set steering
        for steering_motor in self.steering:
            if steering_motor:
                steering_motor.setPosition(steering_angle)
        
        # Set wheel speeds
        for wheel in self.wheels:
            if wheel:
                wheel.setVelocity(speed)

    def run(self):
        """Main AI loop."""
        print("🤖 AI Competitor starting...")
        
        while self.robot.step(self.timestep) != -1:
            # Calculate navigation
            steering_angle = self.calculate_steering_angle()
            speed = self.calculate_speed(steering_angle)
            
            # Apply obstacle avoidance
            steering_angle, speed = self.avoid_obstacles(steering_angle, speed)
            
            # Set car movement
            self.set_car_movement(speed, steering_angle)

# Create and run the AI controller
if __name__ == "__main__":
    controller = AICompetitorController()
    controller.run()