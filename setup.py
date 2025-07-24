#!/usr/bin/env python3
"""
Setup script for Autonomous Driving Demo.

This script helps set up the environment and run initial tests.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True


def install_dependencies():
    """Install required Python packages."""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "models",
        "logs",
        "reports",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ {directory}/")
    
    return True


def check_webots():
    """Check if Webots is available."""
    print("\n🤖 Checking Webots installation...")
    
    # Check for common Webots installation paths
    webots_paths = [
        "/usr/local/webots",
        "/opt/webots",
        "C:\\Program Files\\Webots",
        "C:\\Webots",
        "/Applications/Webots.app"
    ]
    
    webots_found = False
    for path in webots_paths:
        if os.path.exists(path):
            print(f"✅ Webots found at: {path}")
            webots_found = True
            break
    
    if not webots_found:
        print("⚠️  Webots not found in common locations")
        print("   Please install Webots R2023b or later from:")
        print("   https://www.cyberbotics.com/")
    
    return webots_found


def run_basic_tests():
    """Run basic functionality tests."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        from controllers.autonomous_vehicle import AutonomousVehicle
        from training.imitation_learning import ImitationLearningTrainer
        from training.reinforcement_learning import ReinforcementLearningTrainer
        from utils.sensor_processing import StateExtractor
        
        print("✅ All modules import successfully")
        
        # Test component initialization
        vehicle = AutonomousVehicle()
        print("✅ Vehicle controller initialization test passed")
        
        trainer = ImitationLearningTrainer()
        print("✅ Imitation learning trainer test passed")
        
        rl_trainer = ReinforcementLearningTrainer()
        print("✅ RL trainer test passed")
        
        state_extractor = StateExtractor()
        print("✅ State extractor test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def generate_quick_start_guide():
    """Generate a quick start guide."""
    guide_content = """# Autonomous Driving Demo - Quick Start Guide

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
"""
    
    with open("QUICK_START.md", "w") as f:
        f.write(guide_content)
    
    print("✅ Quick start guide generated: QUICK_START.md")


def main():
    """Main setup function."""
    print("🚗 Autonomous Driving Demo Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Continuing with limited functionality...")
    
    # Create directories
    create_directories()
    
    # Check Webots
    check_webots()
    
    # Run tests
    if not run_basic_tests():
        print("⚠️  Some components may not work properly")
    
    # Generate quick start guide
    generate_quick_start_guide()
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo get started:")
    print("  python demo.py --mode comprehensive")
    print("\nFor more options:")
    print("  python demo.py --help")
    print("\nRead QUICK_START.md for detailed instructions")


if __name__ == "__main__":
    main()