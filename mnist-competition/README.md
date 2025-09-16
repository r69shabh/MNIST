# MNIST Digit Recognition Solution

This project provides a comprehensive solution for the MNIST digit recognition competition using multiple machine learning approaches.

## Overview

The MNIST dataset contains 70,000 images of handwritten digits (0-9), each image is 28x28 pixels grayscale. The goal is to correctly classify these digits.

## Files

- `mnist_solution.py` - Complete solution with multiple algorithms (Random Forest, KNN, SVM, Neural Network)
- `mnist_tensorflow_solution.py` - Alternative solution using TensorFlow's built-in MNIST dataset
- `requirements.txt` - Required Python packages
- `README.md` - This file

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the Data

#### Option A: Using Kaggle API (Recommended)
1. Create a Kaggle account and get API credentials
2. Set up Kaggle API following [these instructions](https://github.com/Kaggle/kaggle-api)
3. Download the data:
```bash
kaggle competitions download -c digit-recognizer
unzip digit-recognizer.zip
```

#### Option B: Use TensorFlow's built-in dataset
If you don't have access to the competition data, you can use the TensorFlow solution which uses the built-in MNIST dataset.

## Running the Solution

### Complete Solution (requires competition data)
```bash
python mnist_solution.py
```

### TensorFlow Solution (uses built-in data)
```bash
python mnist_tensorflow_solution.py
```

## Algorithms Implemented

### 1. Random Forest
- Fast training and prediction
- Good baseline performance
- Handles high-dimensional data well

### 2. K-Nearest Neighbors (KNN)
- Simple and interpretable
- Good for comparison
- Can be computationally expensive

### 3. Support Vector Machine (SVM)
- Powerful for classification
- Uses subset of data due to computational constraints
- Good theoretical foundation

### 4. Convolutional Neural Network (CNN)
- State-of-the-art for image classification
- Learns hierarchical features
- Best performance expected

## Model Performance

The solution will train all models and compare their performance on a validation set. Typical expected accuracies:

- Random Forest: ~95-97%
- KNN: ~95-97%
- SVM: ~96-98%
- CNN: ~98-99%+

## Output Files

The solution will generate:
- `digit_distribution.png` - Distribution of digits in training data
- `sample_digits.png` - Visualization of sample digits
- `training_history.png` - Neural network training history
- `confusion_matrix.png` - Confusion matrix for model evaluation
- `submission_*.csv` - Prediction file in competition format

## Key Features

### Data Exploration
- Visualizes digit distribution
- Shows sample images
- Checks for missing values

### Preprocessing
- Normalizes pixel values to [0,1]
- Splits data for validation
- Reshapes data for different algorithms

### Model Training
- Multiple algorithms for comparison
- Validation accuracy reporting
- Model persistence

### Evaluation
- Accuracy metrics
- Confusion matrices
- Classification reports
- Training history visualization

## Competition Submission

The solution creates a submission file in the required format:
```csv
ImageId,Label
1,3
2,7
3,8
...
```

## Tips for Better Performance

1. **Data Augmentation**: Rotate, shift, or zoom images to increase training data
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Hyperparameter Tuning**: Use grid search or random search
4. **Advanced Architectures**: Try deeper CNNs or pre-trained models
5. **Cross-Validation**: Use k-fold cross-validation for more robust evaluation

## Troubleshooting

### Common Issues

1. **Kaggle API not set up**: Follow Kaggle API setup instructions
2. **Memory issues with SVM**: Reduce subset size in `train_svm()` method
3. **TensorFlow installation**: Use `pip install tensorflow` or `conda install tensorflow`
4. **GPU support**: Install `tensorflow-gpu` for faster training

### Performance Tips

- Use GPU for neural network training if available
- Reduce dataset size for faster experimentation
- Use early stopping to prevent overfitting

## Next Steps

After running the basic solution, consider:

1. **Hyperparameter optimization** using GridSearchCV or RandomizedSearchCV
2. **Ensemble methods** combining multiple models
3. **Advanced CNN architectures** like ResNet or EfficientNet
4. **Data augmentation** to improve generalization
5. **Transfer learning** using pre-trained models

## References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Kaggle MNIST Competition](https://www.kaggle.com/c/digit-recognizer)
- [TensorFlow MNIST Tutorial](https://www.tensorflow.org/tutorials/quickstart/beginner)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## License

This project is open source and available under the MIT License.