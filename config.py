"""
Configuration file for Fruit and Vegetable CNN Classifier
"""

# Data Configuration
DATA_CONFIG = {
    'train_path': 'data/processed/train',
    'validation_path': 'data/processed/validation',
    'test_path': 'data/processed/test',
    'image_size': (100, 100),
    'channels': 3,
    'train_split': 0.7,
    'validation_split': 0.15,
    'test_split': 0.15,
}

# Model Configuration
MODEL_CONFIG = {
    'input_shape': (100, 100, 3),
    'num_classes': 131,  # Adjust based on your dataset
    'dropout_rate': 0.5,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'validation_split': 0.2,
    'early_stopping_patience': 5,
    'reduce_lr_patience': 3,
    'reduce_lr_factor': 0.5,
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'horizontal_flip': True,
    'zoom_range': 0.2,
    'fill_mode': 'nearest',
}

# Paths
PATHS = {
    'model_save_path': 'models/best_model.h5',
    'model_checkpoint': 'models/checkpoints/',
    'history_path': 'models/training_history.pkl',
    'logs_path': 'logs/',
}

# Random Seed for Reproducibility
RANDOM_SEED = 42
