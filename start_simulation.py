#!/usr/bin/env python3
"""
🚗 Autonomous Driving Simulation Starter

This script helps you start the simulation with minimal setup.
Run this script and follow the interactive prompts.
"""

import os
import sys
import subprocess
import platform
import time
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("🚗 AUTONOMOUS DRIVING SIMULATION")
    print("=" * 60)
    print("Welcome! This script will help you start the simulation.")
    print()


def check_python():
    """Check Python version."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print("✅ Python", sys.version.split()[0], "detected")
    return True


def check_webots():
    """Check if Webots is installed."""
    print("\n🤖 Checking Webots installation...")
    
    # Common Webots paths
    webots_paths = {
        'Windows': [
            'C:\\Program Files\\Webots\\bin\\webots.exe',
            'C:\\Webots\\bin\\webots.exe'
        ],
        'Darwin': [  # macOS
            '/Applications/Webots.app/Contents/MacOS/webots'
        ],
        'Linux': [
            '/usr/local/bin/webots',
            '/opt/webots/bin/webots',
            '/snap/bin/webots'
        ]
    }
    
    system = platform.system()
    paths_to_check = webots_paths.get(system, [])
    
    # Also check if webots is in PATH
    try:
        result = subprocess.run(['webots', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Webots found in system PATH")
            return 'webots'
    except:
        pass
    
    # Check common installation paths
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ Webots found at: {path}")
            return path
    
    print("❌ Webots not found!")
    print("\n📥 Please install Webots:")
    print("1. Go to: https://cyberbotics.com/")
    print("2. Download Webots R2023b or later")
    print("3. Install and try again")
    return None


def check_dependencies():
    """Check if Python dependencies are installed."""
    print("\n📦 Checking Python dependencies...")
    
    required_packages = ['numpy', 'yaml']  # Core packages only
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        
        # Try to install them
        response = input("Install missing packages? (y/n): ").lower().strip()
        if response == 'y':
            try:
                print("Installing packages...")
                subprocess.run([sys.executable, '-m', 'pip', 'install'] + 
                             missing_packages, check=True)
                print("✅ Packages installed successfully!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install packages")
                print("Please run: pip install " + " ".join(missing_packages))
                return False
        else:
            return False
    
    return True


def select_demo_mode():
    """Let user select demo mode."""
    print("\n🎮 Select Demo Mode:")
    print("1. 🏃 Quick Demo (No Webots required)")
    print("2. 🚗 Basic Webots Simulation")
    print("3. 🧠 AI Training Demo")
    print("4. 📊 Complete Pipeline")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            if choice in ['1', '2', '3', '4']:
                return int(choice)
            else:
                print("Please enter 1, 2, 3, or 4")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            sys.exit(0)


def run_quick_demo():
    """Run the standalone demo."""
    print("\n🏃 Running Quick Demo...")
    print("This demo runs without Webots and shows the system architecture.")
    print("Press Ctrl+C to stop at any time.\n")
    
    try:
        subprocess.run([sys.executable, 'simple_demo.py'])
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except FileNotFoundError:
        print("❌ simple_demo.py not found. Make sure you're in the project directory.")


def run_webots_simulation(webots_path):
    """Run Webots with the lane following world."""
    print("\n🚗 Starting Webots Simulation...")
    
    # Check if world file exists
    world_file = "worlds/lane.wbt"
    if not os.path.exists(world_file):
        print(f"❌ World file not found: {world_file}")
        print("Make sure you're in the project directory.")
        return
    
    print("📂 Opening lane following world...")
    print("🎮 Instructions:")
    print("  - Click the ▶️ Play button in Webots to start")
    print("  - The BMW X5 should start driving autonomously")
    print("  - Press Ctrl+C here to stop this script")
    print()
    
    try:
        if webots_path == 'webots':
            subprocess.run(['webots', world_file])
        else:
            subprocess.run([webots_path, world_file])
    except KeyboardInterrupt:
        print("\n⏹️  Simulation stopped")
    except Exception as e:
        print(f"❌ Error starting Webots: {e}")


def run_ai_demo():
    """Run AI training demonstration."""
    print("\n🧠 Running AI Training Demo...")
    print("This will demonstrate imitation learning and reinforcement learning.")
    print("Note: Requires all dependencies to be installed.\n")
    
    try:
        subprocess.run([sys.executable, 'demo.py', '--mode', 'train'])
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except FileNotFoundError:
        print("❌ demo.py not found. Make sure you're in the project directory.")


def run_complete_pipeline():
    """Run the complete demonstration pipeline."""
    print("\n📊 Running Complete Pipeline...")
    print("This runs the full autonomous driving pipeline:")
    print("- Rule-based driving")
    print("- Data collection") 
    print("- Imitation learning")
    print("- Reinforcement learning")
    print("- Performance comparison\n")
    
    try:
        subprocess.run([sys.executable, 'demo.py', '--mode', 'comprehensive'])
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
    except FileNotFoundError:
        print("❌ demo.py not found. Make sure you're in the project directory.")


def main():
    """Main function."""
    print_banner()
    
    # Check prerequisites
    if not check_python():
        return
    
    webots_path = check_webots()
    
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        return
    
    # Select demo mode
    choice = select_demo_mode()
    
    if choice == 1:
        run_quick_demo()
    elif choice == 2:
        if webots_path:
            run_webots_simulation(webots_path)
        else:
            print("\n❌ Webots required for this mode. Please install Webots first.")
    elif choice == 3:
        run_ai_demo()
    elif choice == 4:
        run_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("🎉 Thank you for trying the Autonomous Driving Simulation!")
    print("📚 For more options, check:")
    print("   - INSTALLATION_GUIDE.md (detailed setup)")
    print("   - QUICK_START.md (usage guide)")
    print("   - demo.py --help (command line options)")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check the installation and try again.")