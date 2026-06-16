"""
CNN Model architecture for Fruit and Vegetable classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from config import MODEL_CONFIG


def build_cnn_model(input_shape=(100, 100, 3), num_classes=131):
    """
    Build a CNN model for fruit and vegetable classification.
    
    Args:
        input_shape: Tuple of input image dimensions (height, width, channels)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Flatten and Dense layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(MODEL_CONFIG['dropout_rate']),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(MODEL_CONFIG['dropout_rate']),
        
        Dense(num_classes, activation='softmax')
    ])
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with appropriate loss and metrics.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    
    Returns:
        Compiled model
    """
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')]
    )
    return model


def get_model(input_shape=(100, 100, 3), num_classes=131, learning_rate=0.001):
    """
    Build and compile a CNN model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    model = build_cnn_model(input_shape, num_classes)
    model = compile_model(model, learning_rate)
    return model


if __name__ == "__main__":
    # Test model creation
    model = get_model()
    model.summary()
