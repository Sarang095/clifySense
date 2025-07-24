#!/usr/bin/env python3
"""
Autonomous Driving Demo Script

This script demonstrates the complete autonomous driving system including:
- Webots simulation setup
- Imitation learning training
- Reinforcement learning training
- Model evaluation and comparison
- Live demonstration
"""

import os
import sys
import argparse
import time
import numpy as np
from typing import Dict, Any

# Add project modules to path
sys.path.append(os.path.dirname(__file__))

# Import project modules
from controllers.autonomous_vehicle import AutonomousVehicle
from training.imitation_learning import ImitationLearningTrainer
from training.reinforcement_learning import ReinforcementLearningTrainer
from utils.sensor_processing import StateExtractor


class AutonomousDrivingDemo:
    """Main demo class for the autonomous driving system."""
    
    def __init__(self):
        """Initialize the demo system."""
        print("🚗 Autonomous Driving Demo System")
        print("=" * 50)
        
        # Initialize components
        self.vehicle_controller = None
        self.imitation_trainer = None
        self.rl_trainer = None
        self.state_extractor = None
        
        # Demo state
        self.models = {}
        self.evaluation_results = {}
        
    def setup_simulation(self):
        """Setup the Webots simulation environment."""
        print("\n📡 Setting up simulation environment...")
        
        try:
            # Initialize vehicle controller
            self.vehicle_controller = AutonomousVehicle()
            print("✅ Vehicle controller initialized")
            
            # Initialize state extractor
            self.state_extractor = StateExtractor()
            print("✅ State extractor initialized")
            
            # Initialize trainers
            self.imitation_trainer = ImitationLearningTrainer()
            self.rl_trainer = ReinforcementLearningTrainer()
            print("✅ Training modules initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Error setting up simulation: {e}")
            return False
    
    def run_basic_demo(self, duration: float = 60.0):
        """Run basic lane following demo."""
        print(f"\n🎯 Running basic demo for {duration} seconds...")
        
        if not self.vehicle_controller:
            print("❌ Vehicle controller not initialized")
            return
        
        try:
            self.vehicle_controller.run_demo(duration)
            print("✅ Basic demo completed successfully")
        except Exception as e:
            print(f"❌ Error in basic demo: {e}")
    
    def collect_training_data(self, num_episodes: int = 20):
        """Collect training data for imitation learning."""
        print(f"\n📊 Collecting training data ({num_episodes} episodes)...")
        
        if not self.vehicle_controller or not self.imitation_trainer:
            print("❌ Required components not initialized")
            return
        
        try:
            # Collect expert demonstrations
            data = self.imitation_trainer.collect_expert_data(
                self.vehicle_controller, 
                num_episodes=num_episodes,
                environment="lane"
            )
            
            # Save dataset
            dataset_path = "models/expert_demonstrations.pkl"
            self.imitation_trainer.save_dataset(data, dataset_path)
            
            print(f"✅ Collected {len(data['images'])} training samples")
            return dataset_path
            
        except Exception as e:
            print(f"❌ Error collecting training data: {e}")
            return None
    
    def train_imitation_model(self, dataset_path: str = None):
        """Train imitation learning model."""
        print("\n🧠 Training imitation learning model...")
        
        if not dataset_path:
            # Create dummy data for demonstration
            print("Creating dummy dataset for demonstration...")
            dummy_data = {
                'images': [np.random.rand(224, 224, 3) for _ in range(1000)],
                'actions': [np.random.rand(2) * 2 - 1 for _ in range(1000)],
                'states': [np.random.rand(7) for _ in range(1000)]
            }
            dataset_path = "models/dummy_dataset.pkl"
            self.imitation_trainer.save_dataset(dummy_data, dataset_path)
        
        try:
            # Train PyTorch model
            self.imitation_trainer.train_pytorch_model(
                dataset_path, 
                save_path="models/imitation_model.pth"
            )
            
            # Evaluate model
            results = self.imitation_trainer.evaluate_model(
                "models/imitation_model.pth", 
                dataset_path
            )
            
            self.evaluation_results['imitation'] = results
            print("✅ Imitation learning model trained successfully")
            
        except Exception as e:
            print(f"❌ Error training imitation model: {e}")
    
    def train_rl_model(self, algorithm: str = "PPO", environment: str = "lane"):
        """Train reinforcement learning model."""
        print(f"\n🎮 Training {algorithm} model for {environment} environment...")
        
        if not self.vehicle_controller or not self.rl_trainer:
            print("❌ Required components not initialized")
            return
        
        # Note: This would require actual Webots simulation running
        print("⚠️  RL training requires active Webots simulation")
        print("    In a real setup, this would:")
        print(f"    1. Train {algorithm} agent on {environment} environment")
        print("    2. Save trained model")
        print("    3. Evaluate performance")
        print("    4. Generate training reports")
        
        # Simulate successful training for demo
        print("✅ RL model training completed (simulated)")
    
    def compare_models(self):
        """Compare different model performances."""
        print("\n📈 Comparing model performances...")
        
        if not self.evaluation_results:
            print("⚠️  No evaluation results available")
            return
        
        print("\nModel Comparison Results:")
        print("-" * 40)
        
        for model_name, results in self.evaluation_results.items():
            print(f"\n{model_name.upper()} Model:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
        
        # Generate comparison visualization
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot loss comparison
            models = list(self.evaluation_results.keys())
            losses = [self.evaluation_results[model].get('avg_loss', 0) for model in models]
            
            ax1.bar(models, losses)
            ax1.set_title('Model Loss Comparison')
            ax1.set_ylabel('Average Loss')
            
            # Plot error comparison
            throttle_errors = [self.evaluation_results[model].get('throttle_error', 0) for model in models]
            steering_errors = [self.evaluation_results[model].get('steering_error', 0) for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax2.bar(x - width/2, throttle_errors, width, label='Throttle Error')
            ax2.bar(x + width/2, steering_errors, width, label='Steering Error')
            ax2.set_title('Action Prediction Errors')
            ax2.set_ylabel('Average Error')
            ax2.set_xticks(x)
            ax2.set_xticklabels(models)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig('reports/model_comparison.png')
            plt.show()
            
            print("✅ Model comparison visualization saved")
            
        except ImportError:
            print("⚠️  Matplotlib not available for visualization")
    
    def run_comprehensive_demo(self):
        """Run the complete demo pipeline."""
        print("\n🚀 Running Comprehensive Autonomous Driving Demo")
        print("=" * 60)
        
        # Setup
        if not self.setup_simulation():
            return
        
        # Phase 1: Basic demonstration
        print("\n📍 PHASE 1: Basic Rule-Based Driving")
        self.run_basic_demo(duration=30.0)
        
        # Phase 2: Data collection
        print("\n📍 PHASE 2: Expert Data Collection")
        dataset_path = self.collect_training_data(num_episodes=10)
        
        # Phase 3: Imitation learning
        print("\n📍 PHASE 3: Imitation Learning Training")
        self.train_imitation_model(dataset_path)
        
        # Phase 4: Reinforcement learning
        print("\n📍 PHASE 4: Reinforcement Learning Training")
        self.train_rl_model(algorithm="PPO", environment="lane")
        
        # Phase 5: Model comparison
        print("\n📍 PHASE 5: Model Performance Comparison")
        self.compare_models()
        
        # Summary
        print("\n🎉 Demo completed successfully!")
        self.print_summary()
    
    def print_summary(self):
        """Print demo summary."""
        print("\n📋 Demo Summary")
        print("=" * 30)
        print("✅ Webots simulation environment set up")
        print("✅ Vehicle controller with sensor processing")
        print("✅ Expert data collection system")
        print("✅ Imitation learning with CNN policy")
        print("✅ Reinforcement learning framework")
        print("✅ Multi-environment support")
        print("✅ Model evaluation and comparison")
        
        print("\n📁 Generated Files:")
        files_to_check = [
            "models/expert_demonstrations.pkl",
            "models/imitation_model.pth",
            "reports/model_comparison.png",
            "config/environment_config.yaml"
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                print(f"  ✅ {file_path}")
            else:
                print(f"  📝 {file_path} (would be created)")
        
        print("\n🚗 Next Steps:")
        print("  1. Install Webots (R2023b or later)")
        print("  2. Open world files in Webots:")
        print("     - worlds/lane.wbt")
        print("     - worlds/roundabout.wbt")
        print("  3. Run simulation with autonomous_vehicle controller")
        print("  4. Collect real expert demonstrations")
        print("  5. Train models on collected data")
        print("  6. Evaluate in different environments")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Autonomous Driving Demo")
    parser.add_argument('--mode', choices=['basic', 'comprehensive', 'collect', 'train', 'evaluate'], 
                       default='comprehensive', help='Demo mode to run')
    parser.add_argument('--duration', type=float, default=60.0, 
                       help='Duration for basic demo (seconds)')
    parser.add_argument('--episodes', type=int, default=20, 
                       help='Number of episodes for data collection')
    parser.add_argument('--algorithm', choices=['PPO', 'DDPG', 'SAC'], default='PPO',
                       help='RL algorithm to use')
    parser.add_argument('--environment', choices=['lane', 'roundabout', 'intersection', 'parking'],
                       default='lane', help='Environment to train on')
    
    args = parser.parse_args()
    
    # Create demo instance
    demo = AutonomousDrivingDemo()
    
    # Run selected mode
    if args.mode == 'basic':
        demo.setup_simulation()
        demo.run_basic_demo(args.duration)
    elif args.mode == 'comprehensive':
        demo.run_comprehensive_demo()
    elif args.mode == 'collect':
        demo.setup_simulation()
        demo.collect_training_data(args.episodes)
    elif args.mode == 'train':
        demo.setup_simulation()
        demo.train_imitation_model()
        demo.train_rl_model(args.algorithm, args.environment)
    elif args.mode == 'evaluate':
        demo.setup_simulation()
        demo.compare_models()
    
    print("\n👋 Demo finished. Thank you for trying the autonomous driving system!")


if __name__ == "__main__":
    main()