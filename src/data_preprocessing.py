"""
Data preprocessing and augmentation utilities
"""

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import DATA_CONFIG, AUGMENTATION_CONFIG, RANDOM_SEED


def load_images_from_directory(directory, image_size=(100, 100)):
    """
    Load images from a directory structure.
    
    Args:
        directory: Path to directory containing subdirectories of each class
        image_size: Target image size (height, width)
    
    Returns:
        images: numpy array of images
        labels: numpy array of class labels
        class_names: list of class names
    """
    images = []
    labels = []
    class_names = []
    
    for idx, class_name in enumerate(sorted(os.listdir(directory))):
        class_path = os.path.join(directory, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        class_names.append(class_name)
        
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img) / 255.0
                
                images.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    return np.array(images), np.array(labels), class_names


def create_data_generators(image_size=(100, 100)):
    """
    Create data generators for training and validation.
    
    Args:
        image_size: Target image size
    
    Returns:
        train_datagen: Training data generator with augmentation
        val_datagen: Validation data generator without augmentation
    """
    train_datagen = ImageDataGenerator(
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode'],
        rescale=1./255
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen


def load_data_from_directories(train_dir, validation_dir, test_dir=None, image_size=(100, 100)):
    """
    Load data from directory structure using ImageDataGenerator.
    
    Args:
        train_dir: Path to training directory
        validation_dir: Path to validation directory
        test_dir: Path to test directory (optional)
        image_size: Target image size
    
    Returns:
        train_generator, validation_generator, (test_generator if test_dir provided)
    """
    train_datagen, val_datagen = create_data_generators(image_size)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=image_size,
        batch_size=DATA_CONFIG['batch_size'],
        class_mode='categorical',
        seed=RANDOM_SEED
    )
    
    validation_generator = val_datagen.flow_from_directory(
        validation_dir,
        target_size=image_size,
        batch_size=DATA_CONFIG['batch_size'],
        class_mode='categorical',
        seed=RANDOM_SEED
    )
    
    if test_dir:
        test_generator = val_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=DATA_CONFIG['batch_size'],
            class_mode='categorical',
            shuffle=False,
            seed=RANDOM_SEED
        )
        return train_generator, validation_generator, test_generator
    
    return train_generator, validation_generator


def normalize_image(image_array):
    """
    Normalize image array to [0, 1] range.
    
    Args:
        image_array: numpy array of pixel values
    
    Returns:
        Normalized image array
    """
    return image_array.astype('float32') / 255.0


def preprocess_image_for_prediction(image_path, image_size=(100, 100)):
    """
    Load and preprocess a single image for prediction.
    
    Args:
        image_path: Path to the image file
        image_size: Target image size
    
    Returns:
        Preprocessed image array ready for model prediction
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(image_size)
        img_array = np.array(img)
        img_array = normalize_image(img_array)
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


if __name__ == "__main__":
    print("Data preprocessing utilities loaded successfully!")
