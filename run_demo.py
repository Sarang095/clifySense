#!/usr/bin/env python3
"""
Simple demo runner for Autonomous Driving System.

This script provides an easy way to run the demo without command line arguments.
"""

import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the demo module
from demo import AutonomousDrivingDemo


def main():
    """Run the autonomous driving demo."""
    print("🚗 Starting Autonomous Driving Demo...")
    print("=" * 50)
    
    # Create and run demo
    demo = AutonomousDrivingDemo()
    
    try:
        # Run comprehensive demo
        demo.run_comprehensive_demo()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Please check your setup and try again")
    
    print("\n👋 Thank you for trying the Autonomous Driving Demo!")


if __name__ == "__main__":
    main()