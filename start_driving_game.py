#!/usr/bin/env python3
"""
Webots Driving Game Launcher
A fun racing game built for Webots simulator!
"""

import os
import sys
import platform

def print_banner():
    """Print game banner."""
    print("🏎️" * 20)
    print("🏁    WEBOTS DRIVING GAME    🏁")
    print("🏎️" * 20)
    print()
    print("🎮 A thrilling racing experience in Webots!")
    print("🏆 Race through checkpoints, collect power-ups, and avoid obstacles!")
    print("🤖 Compete against AI opponents!")
    print()

def print_instructions():
    """Print detailed game instructions."""
    print("📖 GAME INSTRUCTIONS:")
    print("=" * 50)
    print()
    print("🎯 OBJECTIVE:")
    print("  • Complete laps by driving through all 4 checkpoints in order")
    print("  • Collect blue power-ups for speed boosts")
    print("  • Avoid red obstacles to maintain your score")
    print("  • Race against the AI competitor (red car)")
    print()
    print("🎮 CONTROLS:")
    print("  • W or ↑ Arrow    : Accelerate")
    print("  • S or ↓ Arrow    : Brake/Reverse")
    print("  • A or ← Arrow    : Turn Left")
    print("  • D or → Arrow    : Turn Right")
    print("  • SPACEBAR        : Handbrake")
    print("  • R               : Reset car position")
    print("  • ESC             : Quit game")
    print()
    print("🏁 CHECKPOINTS:")
    print("  • 🟢 Green   : Start/Finish Line")
    print("  • 🟡 Yellow  : Checkpoint 2")
    print("  • 🟠 Orange  : Checkpoint 3")
    print("  • 🟣 Purple  : Checkpoint 4")
    print()
    print("💎 SCORING:")
    print("  • +100 points : Pass through checkpoint")
    print("  • +500 points : Complete a lap")
    print("  • +50 points  : Collect power-up")
    print("  • -10 points  : Hit obstacle")
    print()

def check_webots_installation():
    """Check if Webots is likely installed."""
    system = platform.system()
    
    webots_paths = []
    if system == "Windows":
        webots_paths = [
            "C:\\Program Files\\Webots",
            "C:\\Webots",
            "C:\\Program Files (x86)\\Webots"
        ]
    elif system == "Darwin":  # macOS
        webots_paths = [
            "/Applications/Webots.app"
        ]
    else:  # Linux
        webots_paths = [
            "/usr/local/webots",
            "/opt/webots",
            "/snap/webots"
        ]
    
    for path in webots_paths:
        if os.path.exists(path):
            print(f"✅ Webots found at: {path}")
            return True
    
    print("⚠️  Webots installation not found in common locations")
    print("   Please ensure Webots R2023b or later is installed")
    print("   Download from: https://www.cyberbotics.com/")
    return False

def print_setup_instructions():
    """Print setup instructions for Windows."""
    print("🔧 SETUP INSTRUCTIONS FOR WINDOWS:")
    print("=" * 50)
    print()
    print("1️⃣  INSTALL WEBOTS:")
    print("   • Download Webots R2023b or later from:")
    print("     https://www.cyberbotics.com/")
    print("   • Install following the setup wizard")
    print("   • Make sure to install the complete package")
    print()
    print("2️⃣  SETUP THE GAME:")
    print("   • Extract this game folder to a location like:")
    print("     C:\\Users\\YourName\\Documents\\WebotsDrivingGame\\")
    print()
    print("3️⃣  LAUNCH THE GAME:")
    print("   • Start Webots application")
    print("   • File → Open World...")
    print("   • Navigate to the game folder")
    print("   • Open: worlds/driving_game.wbt")
    print("   • Click the ▶️ Play button in Webots")
    print()
    print("4️⃣  GAME CONTROLS:")
    print("   • Use keyboard controls as shown above")
    print("   • The game will display instructions in the console")
    print()

def print_webots_specific_instructions():
    """Print Webots-specific running instructions."""
    print("🎮 HOW TO RUN IN WEBOTS:")
    print("=" * 50)
    print()
    print("📁 File Structure:")
    print("   WebotsDrivingGame/")
    print("   ├── worlds/")
    print("   │   └── driving_game.wbt    ← Open this file")
    print("   └── controllers/")
    print("       ├── driving_game_controller.py")
    print("       └── ai_competitor.py")
    print()
    print("🚀 Steps to Run:")
    print("   1. Launch Webots application")
    print("   2. File → Open World...")
    print("   3. Navigate to this project folder")
    print("   4. Select and open: worlds/driving_game.wbt")
    print("   5. The world will load with cars on the track")
    print("   6. Click the ▶️ Play button (or press Ctrl+2)")
    print("   7. The game will start automatically!")
    print("   8. Use keyboard controls to drive")
    print()
    print("💡 Tips:")
    print("   • Make sure both cars are visible on the track")
    print("   • Check the console output for game messages")
    print("   • If controllers don't start, reload the world")
    print("   • Use the camera follow mode for better view")
    print()

def main():
    """Main launcher function."""
    print_banner()
    print_instructions()
    print()
    check_webots_installation()
    print()
    print_setup_instructions()
    print()
    print_webots_specific_instructions()
    
    print("🎊 READY TO RACE!")
    print("Open 'worlds/driving_game.wbt' in Webots to start playing!")
    print("Have fun! 🏁")

if __name__ == "__main__":
    main()