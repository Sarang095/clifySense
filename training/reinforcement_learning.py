"""
Reinforcement Learning Module for Autonomous Driving.

This module implements PPO/DDPG/SAC algorithms using Stable-Baselines3
to train the agent for various driving scenarios.
"""

import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb


class WebotsDrivingEnv(gym.Env):
    """Custom Gym environment for Webots autonomous driving."""
    
    def __init__(self, vehicle_controller, environment_type="lane", config=None):
        """
        Initialize the driving environment.
        
        Args:
            vehicle_controller: Webots vehicle controller
            environment_type: Type of driving environment (lane, roundabout, etc.)
            config: Configuration dictionary
        """
        super(WebotsDrivingEnv, self).__init__()
        
        self.vehicle_controller = vehicle_controller
        self.environment_type = environment_type
        self.config = config or self._default_config()
        
        # Define action space: [throttle, steering] in range [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Define observation space
        # Image: 224x224x3, Vector state: 7 dimensions
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255, 
                shape=(224, 224, 3), 
                dtype=np.uint8
            ),
            'vector': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(7,),
                dtype=np.float32
            )
        })
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = self.config['environments'][environment_type]['max_episode_steps']
        self.current_state = None
        
        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _default_config(self):
        """Default configuration if none provided."""
        return {
            'environments': {
                'lane': {
                    'max_episode_steps': 1000,
                    'rewards': {
                        'lane_keeping': 1.0,
                        'collision': -100.0,
                        'goal_reached': 50.0,
                        'speed_penalty': -0.1
                    }
                }
            }
        }
    
    def reset(self):
        """Reset the environment to initial state."""
        self.episode_step = 0
        
        # Reset vehicle controller
        state = self.vehicle_controller.reset()
        self.current_state = state
        
        # Return observation in format expected by Stable-Baselines3
        obs = {
            'image': (state['image'] * 255).astype(np.uint8),
            'vector': state['vector_state'].astype(np.float32)
        }
        
        return obs
    
    def step(self, action):
        """Execute one step in the environment."""
        # Execute action through vehicle controller
        state, reward, done, info = self.vehicle_controller.step(
            action, self.environment_type
        )
        
        self.current_state = state
        self.episode_step += 1
        
        # Check episode termination
        if self.episode_step >= self.max_episode_steps:
            done = True
            info['reason'] = 'max_steps_reached'
        
        # Prepare observation
        obs = {
            'image': (state['image'] * 255).astype(np.uint8),
            'vector': state['vector_state'].astype(np.float32)
        }
        
        # Track episode metrics
        if done:
            self.episode_rewards.append(reward)
            self.episode_lengths.append(self.episode_step)
        
        return obs, reward, done, info
    
    def render(self, mode='human'):
        """Render the environment (handled by Webots)."""
        pass
    
    def close(self):
        """Close the environment."""
        pass


class CustomCNNPolicy(ActorCriticCnnPolicy):
    """Custom CNN policy for driving with mixed observations."""
    
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super(CustomCNNPolicy, self).__init__(
            observation_space, action_space, lr_schedule, *args, **kwargs
        )
    
    def _build_mlp_extractor(self):
        """Build the MLP feature extractor."""
        # Custom implementation for mixed image + vector observations
        pass


class ReinforcementLearningTrainer:
    """Main class for reinforcement learning training."""
    
    def __init__(self, config_path: str = "config/environment_config.yaml"):
        """Initialize the RL trainer."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize WandB for experiment tracking
        self.use_wandb = False
        try:
            wandb.init(project="autonomous-driving", config=self.config)
            self.use_wandb = True
            print("WandB initialized for experiment tracking")
        except:
            print("WandB not available, continuing without experiment tracking")
        
        # Model storage
        self.models = {}
        self.training_logs = {}
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'training': {
                'reinforcement_learning': {
                    'algorithm': 'PPO',
                    'learning_rate': 0.0003,
                    'total_timesteps': 1000000,
                    'policy': 'CnnPolicy',
                    'n_steps': 2048,
                    'batch_size': 64,
                    'gamma': 0.99,
                    'gae_lambda': 0.95
                }
            },
            'environments': {
                'lane': {
                    'max_episode_steps': 1000,
                    'rewards': {
                        'lane_keeping': 1.0,
                        'collision': -100.0,
                        'goal_reached': 50.0
                    }
                }
            }
        }
    
    def create_environment(self, vehicle_controller, environment_type: str = "lane"):
        """Create and wrap the training environment."""
        # Create base environment
        env = WebotsDrivingEnv(
            vehicle_controller=vehicle_controller,
            environment_type=environment_type,
            config=self.config
        )
        
        # Monitor wrapper for logging
        env = Monitor(env)
        
        # Create vectorized environment
        vec_env = DummyVecEnv([lambda: env])
        
        # Normalize observations and rewards
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        
        return vec_env
    
    def train_ppo(self, vehicle_controller, environment_type: str = "lane", 
                  model_name: str = "ppo_driving") -> PPO:
        """
        Train PPO agent for autonomous driving.
        
        Args:
            vehicle_controller: Webots vehicle controller
            environment_type: Environment type to train on
            model_name: Name for saving the model
            
        Returns:
            Trained PPO model
        """
        print(f"Training PPO agent for {environment_type} environment...")
        
        # Create environment
        env = self.create_environment(vehicle_controller, environment_type)
        
        # Configure PPO
        rl_config = self.config['training']['reinforcement_learning']
        
        # Create model
        model = PPO(
            policy="MultiInputPolicy",  # For dict observation space
            env=env,
            learning_rate=rl_config['learning_rate'],
            n_steps=rl_config['n_steps'],
            batch_size=rl_config['batch_size'],
            gamma=rl_config['gamma'],
            gae_lambda=rl_config['gae_lambda'],
            verbose=1,
            tensorboard_log="./logs/ppo_tensorboard/",
            device=self.device
        )
        
        # Setup callbacks
        eval_env = self.create_environment(vehicle_controller, environment_type)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/{model_name}_best/",
            log_path=f"./logs/{model_name}/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train the model
        total_timesteps = rl_config['total_timesteps']
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        model_path = f"models/{model_name}_final"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Store model
        self.models[f"{model_name}_{environment_type}"] = model
        
        return model
    
    def train_ddpg(self, vehicle_controller, environment_type: str = "lane",
                   model_name: str = "ddpg_driving") -> DDPG:
        """Train DDPG agent for autonomous driving."""
        print(f"Training DDPG agent for {environment_type} environment...")
        
        # Create environment
        env = self.create_environment(vehicle_controller, environment_type)
        
        # Configure DDPG
        rl_config = self.config['training']['reinforcement_learning']
        
        model = DDPG(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=rl_config['learning_rate'],
            gamma=rl_config['gamma'],
            verbose=1,
            tensorboard_log="./logs/ddpg_tensorboard/",
            device=self.device
        )
        
        # Setup callbacks
        eval_env = self.create_environment(vehicle_controller, environment_type)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/{model_name}_best/",
            log_path=f"./logs/{model_name}/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train
        total_timesteps = rl_config['total_timesteps']
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save model
        model_path = f"models/{model_name}_final"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        self.models[f"{model_name}_{environment_type}"] = model
        
        return model
    
    def train_sac(self, vehicle_controller, environment_type: str = "lane",
                  model_name: str = "sac_driving") -> SAC:
        """Train SAC agent for autonomous driving."""
        print(f"Training SAC agent for {environment_type} environment...")
        
        # Create environment
        env = self.create_environment(vehicle_controller, environment_type)
        
        # Configure SAC
        rl_config = self.config['training']['reinforcement_learning']
        
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=rl_config['learning_rate'],
            gamma=rl_config['gamma'],
            verbose=1,
            tensorboard_log="./logs/sac_tensorboard/",
            device=self.device
        )
        
        # Setup callbacks
        eval_env = self.create_environment(vehicle_controller, environment_type)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"./models/{model_name}_best/",
            log_path=f"./logs/{model_name}/",
            eval_freq=10000,
            deterministic=True,
            render=False
        )
        
        # Train
        total_timesteps = rl_config['total_timesteps']
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save model
        model_path = f"models/{model_name}_final"
        model.save(model_path)
        print(f"Model saved to {model_path}")
        
        self.models[f"{model_name}_{environment_type}"] = model
        
        return model
    
    def evaluate_model(self, model, vehicle_controller, environment_type: str = "lane",
                      n_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate trained model performance.
        
        Args:
            model: Trained RL model
            vehicle_controller: Vehicle controller
            environment_type: Environment type
            n_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Evaluating model on {environment_type} environment...")
        
        # Create evaluation environment
        env = WebotsDrivingEnv(
            vehicle_controller=vehicle_controller,
            environment_type=environment_type,
            config=self.config
        )
        
        episode_rewards = []
        episode_lengths = []
        success_rate = 0
        collision_rate = 0
        
        for episode in tqdm(range(n_episodes)):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Track success and collision rates
            if info.get('reason') == 'goal_reached':
                success_rate += 1
            elif info.get('reason') == 'collision':
                collision_rate += 1
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'success_rate': success_rate / n_episodes,
            'collision_rate': collision_rate / n_episodes,
            'episodes_evaluated': n_episodes
        }
        
        print(f"Evaluation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")
        
        # Log to WandB if available
        if self.use_wandb:
            wandb.log({f"eval_{environment_type}_{key}": value for key, value in metrics.items()})
        
        return metrics
    
    def train_multi_environment(self, vehicle_controller, 
                               environments: List[str] = ["lane", "roundabout", "intersection", "parking"],
                               algorithm: str = "PPO"):
        """
        Train agent on multiple environments sequentially.
        
        Args:
            vehicle_controller: Vehicle controller
            environments: List of environment types
            algorithm: RL algorithm to use
        """
        print(f"Training {algorithm} agent on multiple environments: {environments}")
        
        results = {}
        
        for env_type in environments:
            print(f"\n=== Training on {env_type} environment ===")
            
            # Train on current environment
            if algorithm.upper() == "PPO":
                model = self.train_ppo(vehicle_controller, env_type, f"ppo_{env_type}")
            elif algorithm.upper() == "DDPG":
                model = self.train_ddpg(vehicle_controller, env_type, f"ddpg_{env_type}")
            elif algorithm.upper() == "SAC":
                model = self.train_sac(vehicle_controller, env_type, f"sac_{env_type}")
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Evaluate performance
            eval_results = self.evaluate_model(model, vehicle_controller, env_type)
            results[env_type] = eval_results
            
            print(f"Completed training on {env_type}")
        
        # Generate training report
        self._generate_training_report(results, algorithm)
        
        return results
    
    def _generate_training_report(self, results: Dict[str, Dict[str, float]], algorithm: str):
        """Generate a comprehensive training report."""
        print(f"\n=== {algorithm} Training Report ===")
        
        # Create summary table
        print(f"{'Environment':<15} {'Mean Reward':<12} {'Success Rate':<12} {'Collision Rate':<15}")
        print("-" * 60)
        
        for env_type, metrics in results.items():
            print(f"{env_type:<15} {metrics['mean_reward']:<12.2f} "
                  f"{metrics['success_rate']:<12.2f} {metrics['collision_rate']:<15.2f}")
        
        # Plot results
        self._plot_training_results(results, algorithm)
        
        # Save results
        os.makedirs("reports", exist_ok=True)
        report_path = f"reports/{algorithm.lower()}_training_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(results, f)
        
        print(f"Training report saved to {report_path}")
    
    def _plot_training_results(self, results: Dict[str, Dict[str, float]], algorithm: str):
        """Plot training results."""
        environments = list(results.keys())
        mean_rewards = [results[env]['mean_reward'] for env in environments]
        success_rates = [results[env]['success_rate'] for env in environments]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Mean rewards
        ax1.bar(environments, mean_rewards)
        ax1.set_title(f'{algorithm} - Mean Reward by Environment')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rates
        ax2.bar(environments, success_rates)
        ax2.set_title(f'{algorithm} - Success Rate by Environment')
        ax2.set_ylabel('Success Rate')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'reports/{algorithm.lower()}_results.png')
        plt.show()
    
    def load_model(self, model_path: str, algorithm: str = "PPO"):
        """Load a pre-trained model."""
        if algorithm.upper() == "PPO":
            return PPO.load(model_path)
        elif algorithm.upper() == "DDPG":
            return DDPG.load(model_path)
        elif algorithm.upper() == "SAC":
            return SAC.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def demo_trained_model(self, model, vehicle_controller, environment_type: str = "lane",
                          duration: float = 120.0):
        """
        Run a demo with trained model.
        
        Args:
            model: Trained RL model
            vehicle_controller: Vehicle controller
            environment_type: Environment type
            duration: Demo duration in seconds
        """
        print(f"Running demo with trained model for {duration} seconds...")
        
        # Create environment
        env = WebotsDrivingEnv(
            vehicle_controller=vehicle_controller,
            environment_type=environment_type,
            config=self.config
        )
        
        obs = env.reset()
        start_time = vehicle_controller.robot.getTime() if hasattr(vehicle_controller.robot, 'getTime') else 0
        
        while True:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, done, info = env.step(action)
            
            # Print status
            if env.episode_step % 50 == 0:
                print(f"Step: {env.episode_step}, Reward: {reward:.2f}, "
                      f"Lane Dev: {info.get('lane_deviation', 0.0):.3f}")
            
            # Check termination
            if done:
                print(f"Episode ended: {info.get('reason', 'unknown')}")
                break
            
            # Check time limit
            current_time = vehicle_controller.robot.getTime() if hasattr(vehicle_controller.robot, 'getTime') else env.episode_step * 0.032
            if current_time - start_time > duration:
                break
        
        print("Demo completed!")


def main():
    """Main function for reinforcement learning training."""
    print("Initializing Reinforcement Learning Training...")
    
    # For demonstration purposes, we'll create a mock training scenario
    trainer = ReinforcementLearningTrainer()
    
    print("RL Trainer initialized successfully!")
    print("To use with Webots:")
    print("1. Start Webots with a driving world")
    print("2. Initialize vehicle controller")
    print("3. Call trainer.train_ppo(vehicle_controller) or other algorithms")
    print("4. Evaluate and demo the trained models")


if __name__ == "__main__":
    main()