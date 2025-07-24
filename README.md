# Autonomous Driving Agent - Webots Simulation

An intelligent self-driving agent capable of navigating through diverse traffic scenarios using imitation learning and reinforcement learning in Webots simulation environment.

## Features

- **Multi-Environment Navigation**: Lane following, roundabouts, intersections, and parking
- **Learning Pipeline**: Combination of imitation learning and reinforcement learning
- **Multi-Agent Support**: Scripted vehicles as dynamic obstacles
- **Real-time Visualization**: Webots 3D simulation environment

## Environments

1. **Lane Following** (`lane.wbt`): Straight and multi-lane roads
2. **Roundabout** (`roundabout.wbt`): Circular intersection navigation
3. **Intersection** (`intersection.wbt`): Four-way intersection with traffic rules
4. **Parking** (`parking.wbt`): Precision parking maneuvers

## Quick Start

1. Install Webots (R2023b or later)
2. Install Python dependencies: `pip install -r requirements.txt`
3. Open any world file in Webots
4. Run the simulation

## Technologies

- Webots Simulation Platform
- Python Controllers
- TensorFlow/PyTorch for Neural Networks
- Stable-Baselines3 for Reinforcement Learning
- OpenCV for Computer Vision

## Project Structure

```
├── worlds/              # Webots world files
├── controllers/         # Python controllers
├── models/             # Trained ML models
├── training/           # Training scripts
├── utils/              # Utility functions
└── config/             # Configuration files
```
