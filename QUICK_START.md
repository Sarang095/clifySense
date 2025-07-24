# Autonomous Driving Demo - Quick Start Guide

## Setup Complete! 🎉

Your autonomous driving simulation environment is now ready.

## Quick Demo

Run the basic demo:
```bash
python demo.py --mode basic --duration 30
```

Run the comprehensive demo:
```bash
python demo.py --mode comprehensive
```

## Available Demo Modes

1. **Basic Demo**: Simple rule-based driving
   ```bash
   python demo.py --mode basic --duration 60
   ```

2. **Data Collection**: Collect expert demonstrations
   ```bash
   python demo.py --mode collect --episodes 20
   ```

3. **Training**: Train imitation and RL models
   ```bash
   python demo.py --mode train --algorithm PPO
   ```

4. **Evaluation**: Compare model performances
   ```bash
   python demo.py --mode evaluate
   ```

## Webots Integration

1. Install Webots (R2023b or later)
2. Open world files:
   - `worlds/lane.wbt` - Lane following
   - `worlds/roundabout.wbt` - Roundabout navigation

3. Set controller to `autonomous_vehicle` for the main vehicle

## Project Structure

```
├── controllers/          # Webots controllers
│   ├── autonomous_vehicle.py
│   └── scripted_vehicle.py
├── training/             # ML training modules
│   ├── imitation_learning.py
│   └── reinforcement_learning.py
├── utils/                # Utility functions
│   └── sensor_processing.py
├── worlds/               # Webots world files
├── config/               # Configuration files
├── models/               # Trained models
└── demo.py              # Main demo script
```

## Next Steps

1. Run the demo to see the system in action
2. Modify configurations in `config/environment_config.yaml`
3. Experiment with different environments and algorithms
4. Collect real driving data and train custom models

## Troubleshooting

- **Import errors**: Ensure all dependencies are installed
- **Webots issues**: Check Webots installation and version
- **Training slow**: Consider using GPU acceleration
- **Memory issues**: Reduce batch sizes in configuration

Happy driving! 🚗💨
