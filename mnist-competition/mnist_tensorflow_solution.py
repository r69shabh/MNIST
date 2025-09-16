#!/usr/bin/env python3
"""
MNIST Solution using TensorFlow's built-in dataset
This script demonstrates how to solve MNIST digit recognition
using TensorFlow's built-in MNIST dataset when competition data is not available.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_mnist_data():
    """
    Load MNIST data from TensorFlow
    """
    print("Loading MNIST data from TensorFlow...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocess the data for training
    """
    print("Preprocessing data...")
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    # Convert labels to categorical
    y_train_categorical = keras.utils.to_categorical(y_train, 10)
    y_test_categorical = keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train_categorical, x_test, y_test_categorical, y_test

def visualize_samples(x_train, y_train, num_samples=10):
    """
    Visualize sample digits
    """
    print(f"Visualizing {num_samples} sample digits...")
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_samples):
        axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'Label: {y_train[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_digits_tensorflow.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_label_distribution(y_train):
    """
    Plot the distribution of labels
    """
    unique, counts = np.unique(y_train, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.title('Distribution of Digits in Training Data')
    plt.xlabel('Digit')
    plt.ylabel('Count')
    plt.xticks(range(10))
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('digit_distribution_tensorflow.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_simple_nn_model():
    """
    Create a simple neural network model
    """
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_cnn_model():
    """
    Create a Convolutional Neural Network model
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_test, y_test, epochs=10):
    """
    Train the model and return training history
    """
    print("Training model...")
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """
    Plot training history
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history_tensorflow.png', dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_model(model, x_test, y_test_categorical, y_test_original):
    """
    Evaluate the model and create detailed reports
    """
    print("Evaluating model...")
    
    # Get predictions
    test_loss, test_accuracy = model.evaluate(x_test, y_test_categorical, verbose=0)
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test_original, predicted_classes))
    
    # Confusion matrix
    cm = confusion_matrix(y_test_original, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix_tensorflow.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return test_accuracy, predicted_classes

def show_misclassified_examples(x_test, y_test, predictions, num_examples=10):
    """
    Show examples of misclassified digits
    """
    # Find misclassified examples
    misclassified_indices = np.where(y_test != predictions)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found!")
        return
    
    # Select random misclassified examples
    selected_indices = np.random.choice(misclassified_indices, 
                                      min(num_examples, len(misclassified_indices)), 
                                      replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(selected_indices):
        axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f'True: {y_test[idx]}, Pred: {predictions[idx]}')
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Examples')
    plt.tight_layout()
    plt.savefig('misclassified_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_competition_format_submission(predictions, filename='submission_tensorflow.csv'):
    """
    Create a submission file in Kaggle competition format
    """
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"Submission file saved: {filename}")
    print(f"Submission shape: {submission.shape}")
    print(submission.head(10))

def main():
    """
    Main function to run the complete solution
    """
    print("=== MNIST Digit Recognition with TensorFlow ===")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Visualize samples
    visualize_samples(x_train, y_train)
    
    # Plot label distribution
    plot_label_distribution(y_train)
    
    # Preprocess data
    x_train_processed, y_train_categorical, x_test_processed, y_test_categorical, y_test_original = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # Train Simple Neural Network
    print("\n=== Training Simple Neural Network ===")
    simple_model = create_simple_nn_model()
    print(simple_model.summary())
    
    history_simple = train_model(simple_model, x_train_processed, y_train_categorical, 
                                x_test_processed, y_test_categorical, epochs=5)
    
    simple_accuracy, simple_predictions = evaluate_model(simple_model, x_test_processed, 
                                                        y_test_categorical, y_test_original)
    
    # Train CNN
    print("\n=== Training Convolutional Neural Network ===")
    cnn_model = create_cnn_model()
    print(cnn_model.summary())
    
    history_cnn = train_model(cnn_model, x_train_processed, y_train_categorical, 
                             x_test_processed, y_test_categorical, epochs=10)
    
    plot_training_history(history_cnn)
    
    cnn_accuracy, cnn_predictions = evaluate_model(cnn_model, x_test_processed, 
                                                   y_test_categorical, y_test_original)
    
    # Show misclassified examples
    show_misclassified_examples(x_test_processed, y_test_original, cnn_predictions)
    
    # Compare models
    print("\n=== Model Comparison ===")
    print(f"Simple NN Accuracy: {simple_accuracy:.4f}")
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    
    # Create submission with best model
    best_predictions = cnn_predictions if cnn_accuracy > simple_accuracy else simple_predictions
    best_model_name = "CNN" if cnn_accuracy > simple_accuracy else "Simple NN"
    
    print(f"\nBest model: {best_model_name}")
    create_competition_format_submission(best_predictions)

if __name__ == "__main__":
    main()