# Fruit and Vegetable CNN Classifier

A deep learning project for classifying fruits and vegetables using Convolutional Neural Networks (CNN).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a CNN-based classifier to recognize and categorize different types of fruits and vegetables from images. The model is built using TensorFlow/Keras and trained on the Fruit 360 dataset or custom datasets.

## ✨ Features

- ✅ CNN model for image classification
- ✅ Data preprocessing and augmentation
- ✅ Model training and validation
- ✅ Inference script for predictions
- ✅ Performance evaluation metrics
- ✅ Easy-to-use API

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Setup

```bash
# Clone the repository
git clone https://github.com/ddxhnj/fruit-veg-cnn.git
cd fruit-veg-cnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

### Recommended Datasets

1. **Fruit 360** (Kaggle)
   - 90,000+ images of 131 fruit/vegetable types
   - Download: https://www.kaggle.com/moltean/fruits

2. **Vegetable Image Dataset** (Kaggle)
   - Large collection of vegetable images
   - Download: https://www.kaggle.com/theadityasawant/vegetable-image-dataset

### Dataset Structure

```
data/
├── train/
│   ├── apple/
│   ├── banana/
│   ├── tomato/
│   └── ...
├── validation/
│   ├── apple/
│   ├── banana/
│   └── ...
└── test/
    ├── apple/
    ├── banana/
    └── ...
```

## 💻 Usage

### 1. Prepare Data

```python
python src/data_preprocessing.py --input data/raw --output data/processed
```

### 2. Train Model

```python
python src/train.py --epochs 50 --batch_size 32 --learning_rate 0.001
```

### 3. Make Predictions

```python
python src/predict.py --image path/to/image.jpg --model models/best_model.h5
```

### 4. Evaluate Model

```python
python src/evaluate.py --model models/best_model.h5 --test_data data/test
```

## 🧠 Model Architecture

```
Input Layer (100x100x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (128) + ReLU + Dropout (0.5)
    ↓
Output Layer (num_classes, Softmax)
```

**Model Parameters:**
- Total Parameters: ~1.2M
- Trainable Parameters: ~1.2M
- Input Shape: (100, 100, 3)
- Output Classes: 131 (for Fruit 360)

## 📈 Results

Expected performance metrics on Fruit 360 dataset:
- Training Accuracy: 98%+
- Validation Accuracy: 95%+
- Test Accuracy: 94%+

## 🛠️ Technologies

- **TensorFlow 2.x** - Deep learning framework
- **Keras** - Neural network API
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Pillow** - Image processing
- **Matplotlib** - Visualization
- **scikit-learn** - ML utilities

## 📁 Project Structure

```
fruit-veg-cnn/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
│   └── best_model.h5
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── config.py
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Classifying! 🥕🍎🥒**