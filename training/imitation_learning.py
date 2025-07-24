"""
Imitation Learning Module for Autonomous Driving.

This module implements supervised learning to train the agent to imitate
expert driving behavior using CNN-based models.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt


class DrivingDataset(Dataset):
    """Dataset class for driving data (images and actions)."""
    
    def __init__(self, data_path: str, transform=None):
        """
        Initialize the driving dataset.
        
        Args:
            data_path: Path to the dataset file
            transform: Optional data transformations
        """
        self.data_path = data_path
        self.transform = transform
        self.data = self._load_data()
    
    def _load_data(self) -> Dict[str, List]:
        """Load dataset from file."""
        if os.path.exists(self.data_path):
            with open(self.data_path, 'rb') as f:
                return pickle.load(f)
        else:
            # Return empty dataset if file doesn't exist
            return {'images': [], 'actions': [], 'states': []}
    
    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        image = self.data['images'][idx]
        action = self.data['actions'][idx]
        state = self.data['states'][idx] if 'states' in self.data else np.zeros(7)
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': torch.FloatTensor(image),
            'state': torch.FloatTensor(state),
            'action': torch.FloatTensor(action)
        }


class CNNPolicy(nn.Module):
    """CNN-based policy network for driving."""
    
    def __init__(self, image_shape=(3, 224, 224), state_dim=7, action_dim=2):
        """
        Initialize CNN policy network.
        
        Args:
            image_shape: Shape of input images (channels, height, width)
            state_dim: Dimension of state vector
            action_dim: Dimension of action space (throttle, steering)
        """
        super(CNNPolicy, self).__init__()
        
        # CNN feature extractor for images
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *image_shape)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_size = cnn_output.numel()
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.cnn_output_size + state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Output actions in [-1, 1] range
        )
    
    def forward(self, image, state):
        """Forward pass through the network."""
        # Extract features from image
        cnn_features = self.cnn(image)
        cnn_features = cnn_features.view(cnn_features.size(0), -1)
        
        # Concatenate image features with state
        combined_features = torch.cat([cnn_features, state], dim=1)
        
        # Predict actions
        actions = self.fc(combined_features)
        
        return actions


class TensorFlowCNNPolicy:
    """TensorFlow/Keras implementation of CNN policy."""
    
    def __init__(self, image_shape=(224, 224, 3), state_dim=7, action_dim=2):
        """Initialize TensorFlow CNN policy."""
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the CNN model using Keras."""
        # Image input
        image_input = layers.Input(shape=self.image_shape, name='image_input')
        
        # CNN layers
        x = layers.Conv2D(32, (8, 8), strides=4, activation='relu')(image_input)
        x = layers.Conv2D(64, (4, 4), strides=2, activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # State input
        state_input = layers.Input(shape=(self.state_dim,), name='state_input')
        
        # Combine features
        combined = layers.Concatenate()([x, state_input])
        
        # Fully connected layers
        fc = layers.Dense(512, activation='relu')(combined)
        fc = layers.Dropout(0.3)(fc)
        fc = layers.Dense(256, activation='relu')(fc)
        fc = layers.Dropout(0.2)(fc)
        
        # Output layer
        actions = layers.Dense(self.action_dim, activation='tanh', name='actions')(fc)
        
        # Create model
        model = keras.Model(inputs=[image_input, state_input], outputs=actions)
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss."""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
    
    def train(self, train_data, validation_data=None, epochs=100, batch_size=32):
        """Train the model."""
        callbacks = [
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, image, state):
        """Predict action given image and state."""
        return self.model.predict([image, state])


class ImitationLearningTrainer:
    """Main class for imitation learning training."""
    
    def __init__(self, config_path: str = "config/environment_config.yaml"):
        """Initialize the imitation learning trainer."""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = CNNPolicy().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.config['training']['imitation_learning']['learning_rate']
        )
        self.criterion = nn.MSELoss()
        
        # TensorFlow model for comparison
        self.tf_model = TensorFlowCNNPolicy()
        self.tf_model.compile_model(
            learning_rate=self.config['training']['imitation_learning']['learning_rate']
        )
    
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
                'imitation_learning': {
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'epochs': 100,
                    'validation_split': 0.2
                }
            }
        }
    
    def collect_expert_data(self, vehicle_controller, num_episodes=50, 
                          environment="lane") -> Dict[str, List]:
        """
        Collect expert demonstration data.
        
        Args:
            vehicle_controller: Autonomous vehicle controller
            num_episodes: Number of episodes to collect
            environment: Environment type
            
        Returns:
            Dictionary containing collected data
        """
        print(f"Collecting expert data for {num_episodes} episodes...")
        
        data = {'images': [], 'actions': [], 'states': []}
        
        for episode in tqdm(range(num_episodes)):
            state = vehicle_controller.reset()
            done = False
            
            while not done:
                # Expert policy (simple rule-based for demonstration)
                action = self._expert_policy(state, environment)
                
                # Store data
                data['images'].append(state['image'])
                data['states'].append(state['vector_state'])
                data['actions'].append(action)
                
                # Take step
                state, reward, done, info = vehicle_controller.step(action, environment)
        
        print(f"Collected {len(data['images'])} samples")
        return data
    
    def _expert_policy(self, state: Dict[str, Any], environment: str) -> np.ndarray:
        """
        Simple expert policy for data collection.
        
        Args:
            state: Current state
            environment: Environment type
            
        Returns:
            Expert action
        """
        lane_deviation = state.get('lane_deviation', 0.0)
        obstacles = state.get('obstacles', {})
        
        # Simple lane following with obstacle avoidance
        steering = -lane_deviation * 2.0
        
        if obstacles.get('front', 1.0) < 0.5:
            throttle = -0.5  # Brake
        elif obstacles.get('front', 1.0) < 0.7:
            throttle = 0.1   # Slow down
        else:
            throttle = 0.4   # Normal speed
        
        # Add some steering for obstacle avoidance
        if obstacles.get('left', 1.0) < 0.6:
            steering += 0.3  # Steer right
        elif obstacles.get('right', 1.0) < 0.6:
            steering -= 0.3  # Steer left
        
        # Clamp values
        steering = np.clip(steering, -1.0, 1.0)
        throttle = np.clip(throttle, -1.0, 1.0)
        
        return np.array([throttle, steering])
    
    def save_dataset(self, data: Dict[str, List], filepath: str):
        """Save dataset to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Dataset saved to {filepath}")
    
    def load_dataset(self, filepath: str) -> Dict[str, List]:
        """Load dataset from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        print(f"Dataset loaded from {filepath}")
        return data
    
    def train_pytorch_model(self, dataset_path: str, save_path: str = "models/imitation_pytorch.pth"):
        """Train PyTorch model."""
        print("Training PyTorch model...")
        
        # Load dataset
        dataset = DrivingDataset(dataset_path)
        
        # Split data
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['training']['imitation_learning']['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['training']['imitation_learning']['batch_size']
        )
        
        # Training loop
        epochs = self.config['training']['imitation_learning']['epochs']
        best_val_loss = float('inf')
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                images = batch['image'].to(self.device)
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                
                # Forward pass
                predicted_actions = self.model(images, states)
                loss = self.criterion(predicted_actions, actions)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch in val_loader:
                    images = batch['image'].to(self.device)
                    states = batch['state'].to(self.device)
                    actions = batch['action'].to(self.device)
                    
                    predicted_actions = self.model(images, states)
                    loss = self.criterion(predicted_actions, actions)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_losses, "PyTorch Model")
        
        print(f"Training completed. Best model saved to {save_path}")
    
    def _plot_training_curves(self, train_losses: List[float], val_losses: List[float], title: str):
        """Plot training and validation loss curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{title} - Training Curves')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def evaluate_model(self, model_path: str, dataset_path: str):
        """Evaluate trained model."""
        print("Evaluating model...")
        
        # Load model
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Load dataset
        dataset = DrivingDataset(dataset_path)
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        total_loss = 0.0
        throttle_errors = []
        steering_errors = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                
                predicted_actions = self.model(images, states)
                loss = self.criterion(predicted_actions, actions)
                total_loss += loss.item()
                
                # Calculate individual action errors
                throttle_error = torch.abs(predicted_actions[:, 0] - actions[:, 0])
                steering_error = torch.abs(predicted_actions[:, 1] - actions[:, 1])
                
                throttle_errors.extend(throttle_error.cpu().numpy())
                steering_errors.extend(steering_error.cpu().numpy())
        
        avg_loss = total_loss / len(test_loader)
        avg_throttle_error = np.mean(throttle_errors)
        avg_steering_error = np.mean(steering_errors)
        
        print(f"Evaluation Results:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average Throttle Error: {avg_throttle_error:.4f}")
        print(f"Average Steering Error: {avg_steering_error:.4f}")
        
        return {
            'avg_loss': avg_loss,
            'throttle_error': avg_throttle_error,
            'steering_error': avg_steering_error
        }


def main():
    """Main function for imitation learning training."""
    print("Initializing Imitation Learning Training...")
    
    trainer = ImitationLearningTrainer()
    
    # For demonstration, create some dummy data
    print("Creating dummy dataset for demonstration...")
    dummy_data = {
        'images': [np.random.rand(224, 224, 3) for _ in range(1000)],
        'actions': [np.random.rand(2) * 2 - 1 for _ in range(1000)],  # Random actions in [-1, 1]
        'states': [np.random.rand(7) for _ in range(1000)]
    }
    
    # Save dummy dataset
    dataset_path = "models/dummy_dataset.pkl"
    trainer.save_dataset(dummy_data, dataset_path)
    
    # Train model
    trainer.train_pytorch_model(dataset_path)
    
    # Evaluate model
    trainer.evaluate_model("models/imitation_pytorch.pth", dataset_path)
    
    print("Imitation learning training completed!")


if __name__ == "__main__":
    main()