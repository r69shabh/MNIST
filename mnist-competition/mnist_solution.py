#!/usr/bin/env python3
"""
MNIST Digit Recognition Solution
A comprehensive solution for the MNIST digit recognition competition
using multiple machine learning approaches.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MNISTSolver:
    """
    A comprehensive MNIST digit recognition solver with multiple algorithms
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_data(self, train_path='../nst-contest1/train.csv', test_path='../nst-contest1/test.csv'):
        """
        Load training and test data from CSV files
        """
        print("Loading data...")
        try:
            self.train_data = pd.read_csv(train_path)
            self.test_data = pd.read_csv(test_path)
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure train.csv and test.csv are in the nst-contest1 directory")
            return False
        return True
    
    def explore_data(self):
        """
        Explore and visualize the training data
        """
        print("\n=== Data Exploration ===")
        print(f"Training data shape: {self.train_data.shape}")
        print(f"Test data shape: {self.test_data.shape}")
        
        # Check for missing values
        print(f"Missing values in training data: {self.train_data.isnull().sum().sum()}")
        print(f"Missing values in test data: {self.test_data.isnull().sum().sum()}")
        
        # Label distribution
        print("\nLabel distribution:")
        print(self.train_data['label'].value_counts().sort_index())
        
        # Plot label distribution
        plt.figure(figsize=(10, 6))
        self.train_data['label'].value_counts().sort_index().plot(kind='bar')
        plt.title('Distribution of Digits in Training Data')
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.savefig('digit_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    def visualize_samples(self, num_samples=10):
        """
        Visualize sample digits from the training data
        """
        print(f"\n=== Visualizing {num_samples} Sample Digits ===")
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        for i in range(num_samples):
            # Get pixel values and reshape to 28x28
            pixels = self.train_data.iloc[i, 1:].values.reshape(28, 28)
            label = self.train_data.iloc[i, 0]
            
            axes[i].imshow(pixels, cmap='gray')
            axes[i].set_title(f'Label: {label}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_digits.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """
        Preprocess the data for machine learning
        """
        print("\n=== Preprocessing Data ===")
        
        # Separate features and labels
        self.y_train = self.train_data['label'].values
        self.X_train = self.train_data.drop('label', axis=1).values
        self.X_test = self.test_data.values
        
        # Normalize pixel values to [0, 1]
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        
        # Split training data for validation
        self.X_train_split, self.X_val, self.y_train_split, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        print(f"Training split shape: {self.X_train_split.shape}")
        print(f"Validation split shape: {self.X_val.shape}")
    
    def train_random_forest(self):
        """
        Train a Random Forest classifier
        """
        print("\n=== Training Random Forest ===")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        rf.fit(self.X_train_split, self.y_train_split)
        self.models['random_forest'] = rf
        
        # Evaluate on validation set
        val_pred = rf.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, val_pred)
        print(f"Random Forest Validation Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def train_svm(self):
        """
        Train an SVM classifier (using a subset of data due to computational constraints)
        """
        print("\n=== Training SVM (on subset) ===")
        
        # Use only a subset for SVM due to computational constraints
        subset_size = 5000
        indices = np.random.choice(len(self.X_train_split), subset_size, replace=False)
        X_subset = self.X_train_split[indices]
        y_subset = self.y_train_split[indices]
        
        svm = SVC(kernel='rbf', random_state=42, verbose=True)
        svm.fit(X_subset, y_subset)
        self.models['svm'] = svm
        
        # Evaluate on validation set (subset)
        val_indices = np.random.choice(len(self.X_val), 1000, replace=False)
        X_val_subset = self.X_val[val_indices]
        y_val_subset = self.y_val[val_indices]
        
        val_pred = svm.predict(X_val_subset)
        accuracy = accuracy_score(y_val_subset, val_pred)
        print(f"SVM Validation Accuracy (on subset): {accuracy:.4f}")
        
        return accuracy
    
    def train_knn(self):
        """
        Train a K-Nearest Neighbors classifier
        """
        print("\n=== Training K-Nearest Neighbors ===")
        
        knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        knn.fit(self.X_train_split, self.y_train_split)
        self.models['knn'] = knn
        
        # Evaluate on validation set
        val_pred = knn.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, val_pred)
        print(f"KNN Validation Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def train_neural_network(self):
        """
        Train a neural network using TensorFlow/Keras
        """
        print("\n=== Training Neural Network ===")
        
        # Reshape data for neural network
        X_train_nn = self.X_train_split.reshape(-1, 28, 28, 1)
        X_val_nn = self.X_val.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train_categorical = keras.utils.to_categorical(self.y_train_split, 10)
        y_val_categorical = keras.utils.to_categorical(self.y_val, 10)
        
        # Build the model
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
        
        print(model.summary())
        
        # Train the model
        history = model.fit(
            X_train_nn, y_train_categorical,
            epochs=10,
            batch_size=128,
            validation_data=(X_val_nn, y_val_categorical),
            verbose=1
        )
        
        self.models['neural_network'] = model
        
        # Get final validation accuracy
        val_loss, val_accuracy = model.evaluate(X_val_nn, y_val_categorical, verbose=0)
        print(f"Neural Network Validation Accuracy: {val_accuracy:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return val_accuracy
    
    def make_predictions(self, model_name='neural_network'):
        """
        Make predictions on test data using the specified model
        """
        print(f"\n=== Making Predictions with {model_name} ===")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]
        
        if model_name == 'neural_network':
            # Reshape for neural network
            X_test_reshaped = self.X_test.reshape(-1, 28, 28, 1)
            predictions = model.predict(X_test_reshaped)
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = model.predict(self.X_test)
        
        return predictions
    
    def create_submission(self, predictions, filename='submission.csv'):
        """
        Create submission file in the required format
        """
        print(f"\n=== Creating Submission File: {filename} ===")
        
        submission = pd.DataFrame({
            'ImageId': range(1, len(predictions) + 1),
            'Label': predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission file saved: {filename}")
        print(f"Submission shape: {submission.shape}")
        print(submission.head(10))
        
    def run_complete_solution(self):
        """
        Run the complete solution pipeline
        """
        print("=== MNIST Digit Recognition Solution ===")
        
        # Load data from nst-contest1 folder
        if not self.load_data():
            print("Error loading competition data. Please check that train.csv and test.csv are in ../nst-contest1/ folder")
            return
        
        # Explore data
        self.explore_data()
        
        # Visualize samples
        self.visualize_samples()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train models and compare performance
        accuracies = {}
        
        # Random Forest
        accuracies['Random Forest'] = self.train_random_forest()
        
        # K-Nearest Neighbors
        accuracies['KNN'] = self.train_knn()
        
        # Neural Network
        accuracies['Neural Network'] = self.train_neural_network()
        
        # SVM (optional, takes longer)
        # accuracies['SVM'] = self.train_svm()
        
        # Print comparison
        print("\n=== Model Performance Comparison ===")
        for model, accuracy in accuracies.items():
            print(f"{model}: {accuracy:.4f}")
        
        # Use best model for final predictions
        best_model = max(accuracies, key=accuracies.get)
        print(f"\nBest model: {best_model}")
        
        # Make predictions with best model
        model_name_map = {
            'Random Forest': 'random_forest',
            'KNN': 'knn',
            'Neural Network': 'neural_network',
            'SVM': 'svm'
        }
        
        predictions = self.make_predictions(model_name_map[best_model])
        
        if predictions is not None:
            self.create_submission(predictions, f'submission_{best_model.lower().replace(" ", "_")}.csv')
    
    def create_sample_data(self):
        """
        Create sample data for demonstration if actual data is not available
        """
        print("Creating sample data for demonstration...")
        
        # Create sample training data
        n_samples = 1000
        sample_train = pd.DataFrame()
        sample_train['label'] = np.random.randint(0, 10, n_samples)
        
        # Create random pixel data (normally this would be actual digit images)
        for i in range(784):  # 28x28 = 784 pixels
            sample_train[f'pixel{i}'] = np.random.randint(0, 255, n_samples)
        
        # Create sample test data
        n_test = 100
        sample_test = pd.DataFrame()
        for i in range(784):
            sample_test[f'pixel{i}'] = np.random.randint(0, 255, n_test)
        
        # Save sample data
        sample_train.to_csv('train.csv', index=False)
        sample_test.to_csv('test.csv', index=False)
        
        print("Sample data created: train.csv and test.csv")
        
        # Update data
        self.train_data = sample_train
        self.test_data = sample_test


def main():
    """
    Main function to run the MNIST solution
    """
    solver = MNISTSolver()
    solver.run_complete_solution()


if __name__ == "__main__":
    main()