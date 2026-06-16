"""
Prediction script for Fruit and Vegetable CNN Classifier
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_preprocessing import preprocess_image_for_prediction
from config import DATA_CONFIG, PATHS


def get_class_names_from_generator(train_dir):
    """
    Get class names from training directory.
    
    Args:
        train_dir: Path to training directory
    
    Returns:
        Sorted list of class names
    """
    class_names = sorted(os.listdir(train_dir))
    return [name for name in class_names if os.path.isdir(os.path.join(train_dir, name))]


def predict_image(model, image_path, class_names, image_size=(100, 100), top_k=5):
    """
    Predict the class of an image.
    
    Args:
        model: Trained Keras model
        image_path: Path to image file
        class_names: List of class names
        image_size: Target image size
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions
    """
    # Preprocess image
    img_array = preprocess_image_for_prediction(image_path, image_size)
    
    if img_array is None:
        return None
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_probs = predictions[0]
    
    # Get top-k predictions
    top_k_indices = np.argsort(predicted_probs)[::-1][:top_k]
    
    results = {
        'image': image_path,
        'predictions': []
    }
    
    for idx in top_k_indices:
        results['predictions'].append({
            'class': class_names[idx],
            'probability': float(predicted_probs[idx])
        })
    
    return results


def predict_directory(model, directory, class_names, image_size=(100, 100)):
    """
    Predict classes for all images in a directory.
    
    Args:
        model: Trained Keras model
        directory: Path to directory containing images
        class_names: List of class names
        image_size: Target image size
    
    Returns:
        List of prediction results
    """
    results = []
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        if not os.path.isfile(filepath):
            continue
        
        # Check if file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            continue
        
        print(f"Predicting: {filename}")
        result = predict_image(model, filepath, class_names, image_size)
        
        if result:
            results.append(result)
    
    return results


def main():
    """Main prediction function with argument parsing."""
    parser = argparse.ArgumentParser(description='Predict fruit/vegetable class using trained CNN')
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--directory', type=str, help='Path to directory of images')
    parser.add_argument('--model', type=str, default=PATHS['model_checkpoint'],
                       help='Path to trained model')
    parser.add_argument('--train_dir', type=str, default=DATA_CONFIG['train_path'],
                       help='Path to training directory (for class names)')
    parser.add_argument('--image_size', type=int, default=100,
                       help='Image size (assumes square images)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top predictions to return')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.directory:
        parser.error("Please provide either --image or --directory")
    
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.train_dir):
        print(f"Error: Training directory not found at {args.train_dir}")
        return
    
    print("Loading model...")
    model = load_model(args.model)
    
    print("Getting class names...")
    class_names = get_class_names_from_generator(args.train_dir)
    print(f"Found {len(class_names)} classes")
    
    image_size = (args.image_size, args.image_size)
    
    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            return
        
        print(f"\nPredicting image: {args.image}")
        result = predict_image(model, args.image, class_names, image_size, args.top_k)
        
        if result:
            print(f"\nPredictions for: {result['image']}")
            print("-" * 50)
            for i, pred in enumerate(result['predictions'], 1):
                print(f"{i}. {pred['class']}: {pred['probability']:.4f}")
    
    # Directory prediction
    elif args.directory:
        if not os.path.exists(args.directory):
            print(f"Error: Directory not found at {args.directory}")
            return
        
        print(f"\nPredicting images in: {args.directory}")
        results = predict_directory(model, args.directory, class_names, image_size)
        
        print(f"\n\nPredictions for {len(results)} images:")
        print("=" * 60)
        
        for result in results:
            print(f"\n{result['image']}")
            print("-" * 60)
            for i, pred in enumerate(result['predictions'], 1):
                print(f"  {i}. {pred['class']}: {pred['probability']:.4f}")


if __name__ == "__main__":
    main()
