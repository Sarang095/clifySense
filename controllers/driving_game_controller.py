#!/usr/bin/env python3
"""
Webots Driving Game Controller
A fun racing game with checkpoints, power-ups, and obstacles!

Controls:
- W/A/S/D or Arrow Keys: Drive the car
- Space: Handbrake
- R: Reset car position
- ESC: Quit game

Game Features:
- Checkpoint racing system
- Power-up collection
- Obstacle avoidance
- Score and timer
- AI competitor
"""

import sys
import math
import time
from controller import Robot, Camera, GPS, Compass, DistanceSensor, Keyboard

class DrivingGameController:
    def __init__(self):
        # Initialize the robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Game state
        self.score = 0
        self.lap_time = 0
        self.current_checkpoint = 1
        self.checkpoints_passed = 0
        self.speed_boost = 1.0
        self.boost_timer = 0
        self.game_running = True
        self.lap_count = 0
        self.best_lap_time = float('inf')
        
        # Initialize car components
        self.init_car_components()
        self.init_sensors()
        self.init_keyboard()
        
        # Game start time
        self.start_time = time.time()
        self.lap_start_time = time.time()
        
        print("🏎️  WEBOTS DRIVING GAME STARTED!")
        print("=" * 50)
        print("Controls:")
        print("  W/↑ - Accelerate")
        print("  S/↓ - Brake/Reverse")
        print("  A/← - Turn Left")
        print("  D/→ - Turn Right")
        print("  SPACE - Handbrake")
        print("  R - Reset Position")
        print("  ESC - Quit Game")
        print("=" * 50)
        print("🎯 Objective: Complete laps by passing through all checkpoints!")
        print("🟢 Green = Start/Finish, 🟡 Yellow = Checkpoint 2")
        print("🟠 Orange = Checkpoint 3, 🟣 Purple = Checkpoint 4")
        print("🔵 Blue spheres = Speed boost power-ups")
        print("🔴 Red cylinders = Obstacles (avoid them!)")
        print("=" * 50)

    def init_car_components(self):
        """Initialize car motors and basic components."""
        # Get the four wheels
        self.wheels = []
        wheel_names = ["front left wheel", "front right wheel", 
                      "rear left wheel", "rear right wheel"]
        
        for name in wheel_names:
            wheel = self.robot.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)
        
        # Steering motors
        self.steering = []
        steering_names = ["front left wheel steering motor", "front right wheel steering motor"]
        
        for name in steering_names:
            motor = self.robot.getDevice(name)
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
        
        # Distance sensors for collision detection
        self.distance_sensors = {}
        sensor_names = ["front_sensor", "left_sensor", "right_sensor"]
        
        for name in sensor_names:
            sensor = self.robot.getDevice(name)
            if sensor:
                sensor.enable(self.timestep)
                self.distance_sensors[name] = sensor
        
        # Camera
        self.camera = self.robot.getDevice("front_camera")
        if self.camera:
            self.camera.enable(self.timestep)

    def init_keyboard(self):
        """Initialize keyboard input."""
        self.keyboard = self.robot.getKeyboard()
        self.keyboard.enable(self.timestep)

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

    def check_checkpoints(self):
        """Check if car has passed through checkpoints."""
        pos = self.get_position()
        x, z = pos[0], pos[2]
        
        checkpoint_positions = {
            1: (0, -75),    # Start/Finish (Green)
            2: (75, 0),     # Checkpoint 2 (Yellow)
            3: (0, 75),     # Checkpoint 3 (Orange)
            4: (-75, 0)     # Checkpoint 4 (Purple)
        }
        
        # Check if near current checkpoint (within 8 units)
        target_x, target_z = checkpoint_positions[self.current_checkpoint]
        distance = math.sqrt((x - target_x)**2 + (z - target_z)**2)
        
        if distance < 8:
            self.checkpoints_passed += 1
            self.score += 100
            
            checkpoint_names = {1: "START/FINISH", 2: "CHECKPOINT 2", 
                              3: "CHECKPOINT 3", 4: "CHECKPOINT 4"}
            
            print(f"✅ Passed {checkpoint_names[self.current_checkpoint]}! Score: {self.score}")
            
            # Move to next checkpoint
            self.current_checkpoint += 1
            if self.current_checkpoint > 4:
                self.current_checkpoint = 1
                self.complete_lap()

    def complete_lap(self):
        """Handle lap completion."""
        current_time = time.time()
        lap_time = current_time - self.lap_start_time
        self.lap_count += 1
        
        if lap_time < self.best_lap_time:
            self.best_lap_time = lap_time
            print(f"🏆 NEW BEST LAP TIME: {lap_time:.2f} seconds!")
        
        print(f"🏁 LAP {self.lap_count} COMPLETED!")
        print(f"   Lap Time: {lap_time:.2f}s")
        print(f"   Best Time: {self.best_lap_time:.2f}s")
        print(f"   Total Score: {self.score}")
        
        self.score += 500  # Bonus for completing lap
        self.lap_start_time = current_time

    def check_powerups(self):
        """Check for power-up collection."""
        pos = self.get_position()
        x, z = pos[0], pos[2]
        
        powerup_positions = [(20, -20), (-20, 20)]
        
        for i, (px, pz) in enumerate(powerup_positions):
            distance = math.sqrt((x - px)**2 + (z - pz)**2)
            if distance < 3:  # Close enough to collect
                self.score += 50
                self.speed_boost = 1.5
                self.boost_timer = time.time() + 5  # 5 second boost
                print(f"⚡ SPEED BOOST COLLECTED! (+50 points)")

    def check_obstacles(self):
        """Check for obstacle collisions."""
        sensors = self.distance_sensors
        
        # Check front sensor
        if "front_sensor" in sensors:
            front_distance = sensors["front_sensor"].getValue()
            if front_distance < 2.5:
                self.score = max(0, self.score - 10)  # Lose points for hitting obstacles
                print(f"💥 Obstacle hit! (-10 points) Score: {self.score}")

    def handle_keyboard_input(self):
        """Handle keyboard input for car control."""
        key = self.keyboard.getKey()
        
        # Default values
        steering_angle = 0
        speed = 0
        
        # Movement controls
        if key == ord('W') or key == 315:  # W or Up arrow
            speed = 60 * self.speed_boost
        elif key == ord('S') or key == 317:  # S or Down arrow
            speed = -30 * self.speed_boost
        
        if key == ord('A') or key == 314:  # A or Left arrow
            steering_angle = -0.5
        elif key == ord('D') or key == 316:  # D or Right arrow
            steering_angle = 0.5
        
        # Special controls
        if key == ord(' '):  # Space - Handbrake
            speed = 0
            for wheel in self.wheels:
                wheel.setVelocity(0)
        
        if key == ord('R'):  # R - Reset position
            self.reset_car_position()
        
        if key == 27:  # ESC - Quit
            self.game_running = False
            print("🏁 Game ended by player!")
        
        # Apply movement
        self.set_car_movement(speed, steering_angle)

    def set_car_movement(self, speed, steering_angle):
        """Set car movement based on speed and steering angle."""
        # Set steering
        for steering_motor in self.steering:
            steering_motor.setPosition(steering_angle)
        
        # Set wheel speeds
        for wheel in self.wheels:
            wheel.setVelocity(speed)

    def reset_car_position(self):
        """Reset car to starting position."""
        print("🔄 Resetting car position...")
        # Note: In Webots, you typically need to use supervisor functions
        # to reset positions, but we can at least stop the car
        for wheel in self.wheels:
            wheel.setVelocity(0)

    def update_speed_boost(self):
        """Update speed boost status."""
        current_time = time.time()
        if self.boost_timer > 0 and current_time > self.boost_timer:
            self.speed_boost = 1.0
            self.boost_timer = 0
            print("⚡ Speed boost expired")

    def display_hud(self):
        """Display game information."""
        current_time = time.time()
        game_time = current_time - self.start_time
        
        # Display info every 2 seconds
        if int(game_time) % 2 == 0 and int(game_time) != getattr(self, 'last_display_time', -1):
            self.last_display_time = int(game_time)
            
            pos = self.get_position()
            speed_status = "⚡ BOOSTED" if self.speed_boost > 1.0 else "Normal"
            
            print(f"\n📊 GAME STATUS:")
            print(f"   Score: {self.score}")
            print(f"   Lap: {self.lap_count}")
            print(f"   Next Checkpoint: {self.current_checkpoint}")
            print(f"   Speed: {speed_status}")
            print(f"   Position: ({pos[0]:.1f}, {pos[2]:.1f})")
            print(f"   Game Time: {game_time:.1f}s")
            if self.best_lap_time < float('inf'):
                print(f"   Best Lap: {self.best_lap_time:.2f}s")

    def run(self):
        """Main game loop."""
        print("🚀 Starting game loop...")
        
        while self.robot.step(self.timestep) != -1 and self.game_running:
            # Handle input
            self.handle_keyboard_input()
            
            # Update game state
            self.check_checkpoints()
            self.check_powerups()
            self.check_obstacles()
            self.update_speed_boost()
            
            # Display HUD
            self.display_hud()
        
        # Game ended
        print("\n🏁 GAME OVER!")
        print("=" * 30)
        print(f"Final Score: {self.score}")
        print(f"Laps Completed: {self.lap_count}")
        print(f"Checkpoints Passed: {self.checkpoints_passed}")
        if self.best_lap_time < float('inf'):
            print(f"Best Lap Time: {self.best_lap_time:.2f}s")
        print("Thanks for playing! 🎮")

# Create and run the controller
if __name__ == "__main__":
    controller = DrivingGameController()
    controller.run()