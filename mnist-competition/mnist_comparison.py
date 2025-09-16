#!/usr/bin/env python3
"""
Quick MNIST Comparison Script
This script quickly compares different machine learning algorithms on MNIST data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import time

def load_and_preprocess_data():
    """Load MNIST data and preprocess it"""
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Use a subset for faster testing
    subset_size = 10000
    indices = np.random.choice(len(x_train), subset_size, replace=False)
    x_train_subset = x_train[indices]
    y_train_subset = y_train[indices]
    
    # Flatten and normalize
    x_train_flat = x_train_subset.reshape(subset_size, -1).astype('float32') / 255.0
    x_test_flat = x_test.reshape(len(x_test), -1).astype('float32') / 255.0
    
    # Split for validation
    x_train_split, x_val_split, y_train_split, y_val_split = train_test_split(
        x_train_flat, y_train_subset, test_size=0.2, random_state=42
    )
    
    return (x_train_split, y_train_split), (x_val_split, y_val_split), (x_test_flat, y_test)

def test_algorithm(name, model, x_train, y_train, x_val, y_val):
    """Test a single algorithm and return its performance"""
    print(f"\n=== Testing {name} ===")
    start_time = time.time()
    
    try:
        model.fit(x_train, y_train)
        predictions = model.predict(x_val)
        accuracy = accuracy_score(y_val, predictions)
        training_time = time.time() - start_time
        
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        return {
            'algorithm': name,
            'accuracy': accuracy,
            'training_time': training_time,
            'status': 'success'
        }
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        return {
            'algorithm': name,
            'accuracy': 0.0,
            'training_time': time.time() - start_time,
            'status': 'failed'
        }

def main():
    """Main function to run algorithm comparison"""
    print("=== MNIST Algorithm Comparison ===")
    
    # Load data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Validation data shape: {x_val.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Define algorithms to test
    algorithms = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
        ("SVM (subset)", SVC(kernel='rbf', random_state=42))
    ]
    
    # Test each algorithm
    results = []
    for name, model in algorithms:
        # Use smaller subset for SVM
        if "SVM" in name:
            subset_indices = np.random.choice(len(x_train), 2000, replace=False)
            x_train_small = x_train[subset_indices]
            y_train_small = y_train[subset_indices]
            
            val_indices = np.random.choice(len(x_val), 500, replace=False)
            x_val_small = x_val[val_indices]
            y_val_small = y_val[val_indices]
            
            result = test_algorithm(name, model, x_train_small, y_train_small, x_val_small, y_val_small)
        else:
            result = test_algorithm(name, model, x_train, y_train, x_val, y_val)
        
        results.append(result)
    
    # Display results
    print("\n=== Final Results Summary ===")
    print("-" * 60)
    print(f"{'Algorithm':<25} {'Accuracy':<12} {'Time (s)':<12} {'Status'}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['algorithm']:<25} {result['accuracy']:<12.4f} {result['training_time']:<12.2f} {result['status']}")
    
    # Create results visualization
    successful_results = [r for r in results if r['status'] == 'success']
    if successful_results:
        plt.figure(figsize=(12, 5))
        
        # Accuracy comparison
        plt.subplot(1, 2, 1)
        algorithms_names = [r['algorithm'] for r in successful_results]
        accuracies = [r['accuracy'] for r in successful_results]
        
        bars = plt.bar(range(len(algorithms_names)), accuracies, color='skyblue', edgecolor='navy')
        plt.title('Algorithm Accuracy Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Accuracy')
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        plt.ylim(0.8, 1.0)
        
        # Add accuracy labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        plt.subplot(1, 2, 2)
        times = [r['training_time'] for r in successful_results]
        
        bars = plt.bar(range(len(algorithms_names)), times, color='lightcoral', edgecolor='darkred')
        plt.title('Training Time Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Training Time (seconds)')
        plt.xticks(range(len(algorithms_names)), algorithms_names, rotation=45, ha='right')
        
        # Add time labels on bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01, 
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Find best algorithm
        best_result = max(successful_results, key=lambda x: x['accuracy'])
        print(f"\nBest performing algorithm: {best_result['algorithm']}")
        print(f"Best accuracy: {best_result['accuracy']:.4f}")
        
        # Create a simple submission with best algorithm (using Logistic Regression for speed)
        if any(r['algorithm'] == 'Logistic Regression' and r['status'] == 'success' for r in results):
            print("\nCreating submission with Logistic Regression...")
            lr_model = LogisticRegression(max_iter=1000, random_state=42)
            lr_model.fit(x_train, y_train)
            
            # Make predictions on test set
            test_predictions = lr_model.predict(x_test)
            
            # Create submission file
            submission = pd.DataFrame({
                'ImageId': range(1, len(test_predictions) + 1),
                'Label': test_predictions
            })
            
            submission.to_csv('submission_comparison.csv', index=False)
            print(f"Submission file created: submission_comparison.csv")
            print(f"Test predictions shape: {test_predictions.shape}")
    
    print("\n=== Algorithm Comparison Complete ===")

if __name__ == "__main__":
    main()