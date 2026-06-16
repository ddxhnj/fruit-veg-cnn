"""
Model evaluation script for Fruit and Vegetable CNN Classifier
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import tensorflow as tf
from tensorflow.keras.models import load_model

from data_preprocessing import load_data_from_directories
from config import DATA_CONFIG, PATHS


def evaluate_model(model, test_generator, model_name="Model"):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        model_name: Name of model for display
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    # Get predictions and true labels
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    
    # Calculate metrics
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    f1 = f1_score(true_classes, predicted_classes, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'predictions': predictions
    }
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}\n")
    
    return metrics


def plot_confusion_matrix(true_classes, predicted_classes, class_names, 
                         save_path='models/confusion_matrix.png', figsize=(20, 16)):
    """
    Plot and save confusion matrix.
    
    Args:
        true_classes: Array of true class labels
        predicted_classes: Array of predicted class labels
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size for the plot
    """
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_class_distribution(true_classes, class_names, 
                           save_path='models/class_distribution.png'):
    """
    Plot and save class distribution.
    
    Args:
        true_classes: Array of true class labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    unique, counts = np.unique(true_classes, return_counts=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(unique)), counts)
    plt.xticks(range(len(unique)), [class_names[i] for i in unique], rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution in Test Set')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to {save_path}")
    plt.close()


def plot_per_class_metrics(true_classes, predicted_classes, class_names,
                          save_path='models/per_class_metrics.png', figsize=(14, 8)):
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        true_classes: Array of true class labels
        predicted_classes: Array of predicted class labels
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Calculate per-class metrics
    report = classification_report(true_classes, predicted_classes, 
                                  output_dict=True, zero_division=0)
    
    classes_to_plot = min(len(class_names), 30)  # Limit to first 30 classes for visibility
    
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(classes_to_plot):
        if str(i) in report:
            precisions.append(report[str(i)]['precision'])
            recalls.append(report[str(i)]['recall'])
            f1s.append(report[str(i)]['f1-score'])
    
    x = np.arange(len(precisions))
    width = 0.25
    
    plt.figure(figsize=figsize)
    plt.bar(x - width, precisions, width, label='Precision')
    plt.bar(x, recalls, width, label='Recall')
    plt.bar(x + width, f1s, width, label='F1-Score')
    
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('Per-Class Metrics (First 30 Classes)')
    plt.xticks(x, [class_names[i] for i in range(classes_to_plot)], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics plot saved to {save_path}")
    plt.close()


def generate_classification_report(true_classes, predicted_classes, class_names,
                                  save_path='models/classification_report.txt'):
    """
    Generate and save detailed classification report.
    
    Args:
        true_classes: Array of true class labels
        predicted_classes: Array of predicted class labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(true_classes, predicted_classes,
                                  target_names=class_names, zero_division=0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed Classification Report:")
    print(report)
    print(f"\nClassification report saved to {save_path}")


def main():
    """Main evaluation function with argument parsing."""
    parser = argparse.ArgumentParser(description='Evaluate Fruit and Vegetable CNN Classifier')
    parser.add_argument('--test_dir', default=DATA_CONFIG['test_path'],
                       help='Path to test directory')
    parser.add_argument('--model', type=str, default=PATHS['model_checkpoint'],
                       help='Path to trained model')
    parser.add_argument('--image_size', type=int, default=100,
                       help='Image size (assumes square images)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory not found at {args.test_dir}")
        return
    
    print("Loading model...")
    model = load_model(args.model)
    
    print("Loading test data...")
    _, test_generator = load_data_from_directories(
        args.test_dir,
        args.test_dir,  # Use test_dir as both train and validation for loading
        image_size=(args.image_size, args.image_size)
    )
    
    # Get class names
    class_names = sorted(os.listdir(args.test_dir))
    class_names = [name for name in class_names if os.path.isdir(os.path.join(args.test_dir, name))]
    
    # Evaluate model
    metrics = evaluate_model(model, test_generator, model_name="Fruit and Vegetable Classifier")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    plot_confusion_matrix(
        metrics['true_classes'],
        metrics['predicted_classes'],
        class_names,
        save_path='models/confusion_matrix.png'
    )
    
    plot_class_distribution(
        metrics['true_classes'],
        class_names,
        save_path='models/class_distribution.png'
    )
    
    plot_per_class_metrics(
        metrics['true_classes'],
        metrics['predicted_classes'],
        class_names,
        save_path='models/per_class_metrics.png'
    )
    
    generate_classification_report(
        metrics['true_classes'],
        metrics['predicted_classes'],
        class_names,
        save_path='models/classification_report.txt'
    )
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
