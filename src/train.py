"""
Training script for Fruit and Vegetable CNN Classifier
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from datetime import datetime

from model import get_model
from data_preprocessing import load_data_from_directories
from config import TRAINING_CONFIG, PATHS, RANDOM_SEED, MODEL_CONFIG, DATA_CONFIG


def create_callbacks(model_checkpoint_path, logs_path):
    """
    Create training callbacks.
    
    Args:
        model_checkpoint_path: Path to save best model
        logs_path: Path to save TensorBoard logs
    
    Returns:
        List of callbacks
    """
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=TRAINING_CONFIG['reduce_lr_factor'],
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=logs_path,
            histogram_freq=1
        )
    ]
    
    return callbacks


def plot_training_history(history, save_path='models/training_history.png'):
    """
    Plot and save training history.
    
    Args:
        history: Training history object
        save_path: Path to save the plot
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train_model(train_dir, validation_dir, epochs=None, batch_size=None, 
                learning_rate=None, image_size=(100, 100)):
    """
    Train the CNN model.
    
    Args:
        train_dir: Path to training directory
        validation_dir: Path to validation directory
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Target image size
    
    Returns:
        Trained model and training history
    """
    # Use defaults if not provided
    epochs = epochs or TRAINING_CONFIG['epochs']
    batch_size = batch_size or TRAINING_CONFIG['batch_size']
    learning_rate = learning_rate or TRAINING_CONFIG['learning_rate']
    
    print("Loading data generators...")
    train_generator, validation_generator = load_data_from_directories(
        train_dir, validation_dir, image_size=image_size
    )
    
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    print("Building model...")
    model = get_model(
        input_shape=(image_size[0], image_size[1], 3),
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    
    model.summary()
    
    print("Creating callbacks...")
    callbacks = create_callbacks(PATHS['model_checkpoint'], PATHS['logs_path'])
    
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training history
    os.makedirs(os.path.dirname(PATHS['history_path']), exist_ok=True)
    with open(PATHS['history_path'], 'wb') as f:
        pickle.dump(history.history, f)
    
    print(f"Model saved to {PATHS['model_checkpoint']}")
    print(f"Training history saved to {PATHS['history_path']}")
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def main():
    """Main training function with argument parsing."""
    parser = argparse.ArgumentParser(description='Train Fruit and Vegetable CNN Classifier')
    parser.add_argument('--train_dir', default=DATA_CONFIG['train_path'], 
                       help='Path to training directory')
    parser.add_argument('--val_dir', default=DATA_CONFIG['validation_path'],
                       help='Path to validation directory')
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=TRAINING_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--image_size', type=int, default=100,
                       help='Image size (assumes square images)')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Train model
    model, history = train_model(
        train_dir=args.train_dir,
        validation_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=(args.image_size, args.image_size)
    )
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
