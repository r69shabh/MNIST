#!/usr/bin/env python3
"""
MNIST Competition Solution with Real Data
This script uses the actual competition data from the nst-contest1 folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class MNISTCompetitionSolver:
    """
    MNIST Competition Solver using real competition data
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.models = {}
        
    def load_competition_data(self):
        """
        Load the actual competition data
        """
        print("Loading competition data...")
        try:
            # Load data from nst-contest1 folder
            self.train_data = pd.read_csv('../nst-contest1/train.csv')
            self.test_data = pd.read_csv('../nst-contest1/test.csv')
            
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            
            # Display basic info
            print(f"Training data columns: {list(self.train_data.columns[:5])}... (showing first 5)")
            print(f"Test data columns: {list(self.test_data.columns[:5])}... (showing first 5)")
            
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure train.csv and test.csv are in the ../nst-contest1/ directory")
            return False
    
    def explore_data(self):
        """
        Explore the competition data
        """
        print("\n=== Data Exploration ===")
        
        # Check for missing values
        print(f"Missing values in training data: {self.train_data.isnull().sum().sum()}")
        print(f"Missing values in test data: {self.test_data.isnull().sum().sum()}")
        
        # Label distribution
        print("\nLabel distribution:")
        label_counts = self.train_data['label'].value_counts().sort_index()
        print(label_counts)
        
        # Plot label distribution
        plt.figure(figsize=(10, 6))
        label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title('Distribution of Digits in Competition Training Data')
        plt.xlabel('Digit')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('competition_digit_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Basic statistics
        print(f"\nData statistics:")
        print(f"Total training samples: {len(self.train_data)}")
        print(f"Total test samples: {len(self.test_data)}")
        print(f"Number of features: {self.train_data.shape[1] - 1}")  # -1 for label column
    
    def visualize_sample_digits(self, num_samples=10):
        """
        Visualize sample digits from the competition data
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
        plt.savefig('competition_sample_digits.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def preprocess_data(self):
        """
        Preprocess the competition data
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
        Train Random Forest on competition data
        """
        print("\n=== Training Random Forest ===")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        start_time = time.time()
        rf.fit(self.X_train_split, self.y_train_split)
        training_time = time.time() - start_time
        
        self.models['random_forest'] = rf
        
        # Evaluate on validation set
        val_pred = rf.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, val_pred)
        print(f"Random Forest Validation Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        return accuracy
    
    def train_cnn(self):
        """
        Train CNN on competition data
        """
        print("\n=== Training CNN ===")
        
        # Reshape data for CNN
        X_train_cnn = self.X_train_split.reshape(-1, 28, 28, 1)
        X_val_cnn = self.X_val.reshape(-1, 28, 28, 1)
        
        # Convert labels to categorical
        y_train_categorical = keras.utils.to_categorical(self.y_train_split, 10)
        y_val_categorical = keras.utils.to_categorical(self.y_val, 10)
        
        # Build CNN model
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
        
        print("CNN Architecture:")
        model.summary()
        
        # Train the model
        start_time = time.time()
        history = model.fit(
            X_train_cnn, y_train_categorical,
            epochs=10,
            batch_size=128,
            validation_data=(X_val_cnn, y_val_categorical),
            verbose=1
        )
        training_time = time.time() - start_time
        
        self.models['cnn'] = model
        
        # Get validation accuracy
        val_loss, val_accuracy = model.evaluate(X_val_cnn, y_val_categorical, verbose=0)
        print(f"CNN Validation Accuracy: {val_accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('CNN Model Accuracy (Competition Data)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('CNN Model Loss (Competition Data)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('competition_cnn_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return val_accuracy
    
    def train_logistic_regression(self):
        """
        Train Logistic Regression
        """
        print("\n=== Training Logistic Regression ===")
        
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        
        start_time = time.time()
        lr.fit(self.X_train_split, self.y_train_split)
        training_time = time.time() - start_time
        
        self.models['logistic_regression'] = lr
        
        # Evaluate
        val_pred = lr.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, val_pred)
        print(f"Logistic Regression Validation Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        return accuracy
    
    def make_predictions(self, model_name='cnn'):
        """
        Make predictions on test data
        """
        print(f"\n=== Making Predictions with {model_name} ===")
        
        if model_name not in self.models:
            print(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            return None
        
        model = self.models[model_name]
        
        if model_name == 'cnn':
            # Reshape for CNN
            X_test_reshaped = self.X_test.reshape(-1, 28, 28, 1)
            predictions = model.predict(X_test_reshaped)
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = model.predict(self.X_test)
        
        return predictions
    
    def create_submission(self, predictions, filename='competition_submission.csv'):
        """
        Create submission file for competition
        """
        print(f"\n=== Creating Submission File: {filename} ===")
        
        submission = pd.DataFrame({
            'ImageId': range(1, len(predictions) + 1),
            'Label': predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission file saved: {filename}")
        print(f"Submission shape: {submission.shape}")
        print("\nFirst 10 predictions:")
        print(submission.head(10))
        
        return submission
    
    def run_complete_solution(self):
        """
        Run the complete competition solution
        """
        print("=== MNIST Competition Solution with Real Data ===")
        
        # Load competition data
        if not self.load_competition_data():
            return
        
        # Explore data
        self.explore_data()
        
        # Visualize samples
        self.visualize_sample_digits()
        
        # Preprocess
        self.preprocess_data()
        
        # Train models
        accuracies = {}
        
        # Quick models first
        accuracies['Logistic Regression'] = self.train_logistic_regression()
        accuracies['Random Forest'] = self.train_random_forest()
        
        # CNN (takes longer)
        accuracies['CNN'] = self.train_cnn()
        
        # Compare results
        print("\n=== Model Performance Comparison ===")
        print("-" * 50)
        for model, accuracy in accuracies.items():
            print(f"{model:<20}: {accuracy:.4f}")
        
        # Use best model for final submission
        best_model = max(accuracies, key=accuracies.get)
        print(f"\nBest model: {best_model} ({accuracies[best_model]:.4f})")
        
        # Create predictions with best model
        model_map = {
            'Logistic Regression': 'logistic_regression',
            'Random Forest': 'random_forest',
            'CNN': 'cnn'
        }
        
        best_model_key = model_map[best_model]
        predictions = self.make_predictions(best_model_key)
        
        if predictions is not None:
            submission = self.create_submission(predictions, f'submission_{best_model_key}_competition.csv')
            
            # Also create submission with CNN if it's not the best
            if best_model_key != 'cnn' and 'cnn' in self.models:
                cnn_predictions = self.make_predictions('cnn')
                if cnn_predictions is not None:
                    self.create_submission(cnn_predictions, 'submission_cnn_competition.csv')
        
        print("\n=== Competition Solution Complete! ===")
        print("Ready to submit to competition!")

def main():
    """
    Main function
    """
    solver = MNISTCompetitionSolver()
    solver.run_complete_solution()

if __name__ == "__main__":
    main()