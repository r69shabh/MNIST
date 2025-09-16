#!/usr/bin/env python3
"""
Enhanced MNIST Solution - Target: >99.5% Accuracy
Advanced techniques to push beyond 99.5% accuracy on competition data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
import time
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EnhancedMNISTSolver:
    """
    Enhanced MNIST solver targeting >99.5% accuracy
    """
    
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.models = {}
        
    def load_competition_data(self):
        """Load the competition data"""
        print("Loading competition data...")
        try:
            self.train_data = pd.read_csv('../nst-contest1/train.csv')
            self.test_data = pd.read_csv('../nst-contest1/test.csv')
            
            print(f"Training data shape: {self.train_data.shape}")
            print(f"Test data shape: {self.test_data.shape}")
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
    
    def advanced_preprocess_data(self):
        """
        Advanced preprocessing with normalization and reshaping
        """
        print("\n=== Advanced Data Preprocessing ===")
        
        # Separate features and labels
        self.y_train = self.train_data['label'].values
        self.X_train = self.train_data.drop('label', axis=1).values
        self.X_test = self.test_data.values
        
        # Normalize to [0, 1] range
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        # Reshape for CNN (add channel dimension)
        self.X_train = self.X_train.reshape(-1, 28, 28, 1)
        self.X_test = self.X_test.reshape(-1, 28, 28, 1)
        
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        
        # Stratified split for validation
        self.X_train_split, self.X_val, self.y_train_split, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.15, random_state=42, stratify=self.y_train
        )
        
        # Convert to categorical
        self.y_train_categorical = keras.utils.to_categorical(self.y_train_split, 10)
        self.y_val_categorical = keras.utils.to_categorical(self.y_val, 10)
        
        print(f"Training split shape: {self.X_train_split.shape}")
        print(f"Validation split shape: {self.X_val.shape}")
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator
        """
        print("Setting up data augmentation...")
        
        datagen = ImageDataGenerator(
            rotation_range=10,          # Random rotation ±10 degrees
            width_shift_range=0.1,      # Random horizontal shift
            height_shift_range=0.1,     # Random vertical shift
            zoom_range=0.1,             # Random zoom
            shear_range=0.1,            # Random shear transformation
            fill_mode='nearest'         # Fill pixels for transformations
        )
        
        datagen.fit(self.X_train_split)
        return datagen
    
    def create_enhanced_cnn(self):
        """
        Create an enhanced CNN architecture for >99.5% accuracy
        """
        print("\n=== Building Enhanced CNN Architecture ===")
        
        model = keras.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Use advanced optimizer
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Enhanced CNN Architecture:")
        model.summary()
        
        return model
    
    def create_callbacks(self):
        """
        Create training callbacks for better performance
        """
        callbacks_list = [
            # Reduce learning rate when accuracy plateaus
            callbacks.ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Early stopping to prevent overfitting
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            callbacks.ModelCheckpoint(
                'best_enhanced_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train_enhanced_cnn(self, use_augmentation=True):
        """
        Train the enhanced CNN with advanced techniques
        """
        print("\n=== Training Enhanced CNN ===")
        
        model = self.create_enhanced_cnn()
        callbacks_list = self.create_callbacks()
        
        start_time = time.time()
        
        if use_augmentation:
            print("Training with data augmentation...")
            datagen = self.create_data_augmentation()
            
            # Calculate steps per epoch
            batch_size = 128
            steps_per_epoch = len(self.X_train_split) // batch_size
            
            history = model.fit(
                datagen.flow(self.X_train_split, self.y_train_categorical, batch_size=batch_size),
                steps_per_epoch=steps_per_epoch,
                epochs=50,
                validation_data=(self.X_val, self.y_val_categorical),
                callbacks=callbacks_list,
                verbose=1
            )
        else:
            print("Training without data augmentation...")
            history = model.fit(
                self.X_train_split, self.y_train_categorical,
                batch_size=128,
                epochs=30,
                validation_data=(self.X_val, self.y_val_categorical),
                callbacks=callbacks_list,
                verbose=1
            )
        
        training_time = time.time() - start_time
        
        # Get final validation accuracy
        val_loss, val_accuracy = model.evaluate(self.X_val, self.y_val_categorical, verbose=0)
        
        print(f"\nEnhanced CNN Results:")
        print(f"Final Validation Accuracy: {val_accuracy:.4f}")
        print(f"Training Time: {training_time:.2f} seconds")
        
        self.models['enhanced_cnn'] = model
        
        # Plot training history
        self.plot_enhanced_training_history(history)
        
        return val_accuracy, history
    
    def plot_enhanced_training_history(self, history):
        """
        Plot enhanced training history
        """
        plt.figure(figsize=(15, 5))
        
        # Accuracy plot
        plt.subplot(1, 3, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        plt.title('Enhanced Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss plot
        plt.subplot(1, 3, 2)
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Enhanced Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        plt.subplot(1, 3, 3)
        if 'lr' in history.history:
            plt.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='orange')
            plt.title('Learning Rate Schedule')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.legend()
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Learning Rate\nNot Tracked', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig('enhanced_cnn_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_ensemble_model(self):
        """
        Create an ensemble of multiple models for even higher accuracy
        """
        print("\n=== Creating Ensemble Model ===")
        
        # Create multiple CNN architectures
        models = []
        
        # Model 1: Enhanced CNN
        model1 = self.create_enhanced_cnn()
        models.append(('Enhanced_CNN_1', model1))
        
        # Model 2: Different architecture
        model2 = keras.Sequential([
            layers.Conv2D(64, (5, 5), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        model2.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        models.append(('Enhanced_CNN_2', model2))
        
        return models
    
    def train_ensemble(self):
        """
        Train ensemble of models
        """
        print("\n=== Training Ensemble Models ===")
        
        ensemble_models = self.create_ensemble_model()
        trained_models = []
        
        for name, model in ensemble_models:
            print(f"\nTraining {name}...")
            
            callbacks_list = [
                callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-7),
                callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
            ]
            
            history = model.fit(
                self.X_train_split, self.y_train_categorical,
                batch_size=128,
                epochs=20,
                validation_data=(self.X_val, self.y_val_categorical),
                callbacks=callbacks_list,
                verbose=0
            )
            
            val_loss, val_accuracy = model.evaluate(self.X_val, self.y_val_categorical, verbose=0)
            print(f"{name} Validation Accuracy: {val_accuracy:.4f}")
            
            trained_models.append((name, model, val_accuracy))
        
        self.ensemble_models = trained_models
        return trained_models
    
    def make_ensemble_predictions(self):
        """
        Make predictions using ensemble averaging
        """
        print("\n=== Making Ensemble Predictions ===")
        
        if not hasattr(self, 'ensemble_models'):
            print("No ensemble models found. Training ensemble first...")
            self.train_ensemble()
        
        # Get predictions from all models
        all_predictions = []
        
        for name, model, accuracy in self.ensemble_models:
            predictions = model.predict(self.X_test, verbose=0)
            all_predictions.append(predictions)
            print(f"{name} predictions completed (accuracy: {accuracy:.4f})")
        
        # Average predictions
        ensemble_predictions = np.mean(all_predictions, axis=0)
        final_predictions = np.argmax(ensemble_predictions, axis=1)
        
        return final_predictions
    
    def detailed_evaluation(self, model):
        """
        Perform detailed evaluation of the model
        """
        print("\n=== Detailed Model Evaluation ===")
        
        # Predictions on validation set
        val_predictions = model.predict(self.X_val, verbose=0)
        val_pred_classes = np.argmax(val_predictions, axis=1)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_val, val_pred_classes))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_val, val_pred_classes)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix - Enhanced CNN')
        plt.colorbar()
        tick_marks = np.arange(10)
        plt.xticks(tick_marks, range(10))
        plt.yticks(tick_marks, range(10))
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('enhanced_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Per-class accuracy
        class_accuracies = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class Accuracies:")
        for i, acc in enumerate(class_accuracies):
            print(f"Digit {i}: {acc:.4f}")
    
    def create_submission(self, predictions, filename='enhanced_submission.csv'):
        """
        Create enhanced submission file
        """
        print(f"\n=== Creating Enhanced Submission: {filename} ===")
        
        submission = pd.DataFrame({
            'ImageId': range(1, len(predictions) + 1),
            'Label': predictions
        })
        
        submission.to_csv(filename, index=False)
        print(f"Submission file saved: {filename}")
        print(f"Submission shape: {submission.shape}")
        
        # Analyze predictions
        label_dist = submission['Label'].value_counts().sort_index()
        print("\nPredicted label distribution:")
        for digit, count in label_dist.items():
            percentage = (count / len(submission)) * 100
            print(f"Digit {digit}: {count:,} ({percentage:.1f}%)")
        
        return submission
    
    def run_enhanced_solution(self):
        """
        Run the complete enhanced solution targeting >99.5% accuracy
        """
        print("="*70)
        print("  ENHANCED MNIST SOLUTION - TARGET: >99.5% ACCURACY")
        print("="*70)
        
        # Load data
        if not self.load_competition_data():
            return
        
        # Advanced preprocessing
        self.advanced_preprocess_data()
        
        # Train enhanced CNN with data augmentation
        print("\n🚀 Training Enhanced CNN with Data Augmentation...")
        accuracy, history = self.train_enhanced_cnn(use_augmentation=True)
        
        if accuracy > 0.995:
            print(f"\n🎉 SUCCESS! Achieved {accuracy:.4f} (>{99.5:.1f}%) accuracy!")
        else:
            print(f"\n📈 Current accuracy: {accuracy:.4f}, targeting >99.5%")
            print("Trying ensemble approach...")
            
            # Try ensemble for even higher accuracy
            ensemble_models = self.train_ensemble()
            
            # Evaluate ensemble on validation set
            ensemble_val_preds = []
            for name, model, acc in ensemble_models:
                preds = model.predict(self.X_val, verbose=0)
                ensemble_val_preds.append(preds)
            
            ensemble_val_avg = np.mean(ensemble_val_preds, axis=0)
            ensemble_val_classes = np.argmax(ensemble_val_avg, axis=1)
            ensemble_accuracy = accuracy_score(self.y_val, ensemble_val_classes)
            
            print(f"Ensemble Validation Accuracy: {ensemble_accuracy:.4f}")
            accuracy = max(accuracy, ensemble_accuracy)
        
        # Detailed evaluation
        best_model = self.models['enhanced_cnn']
        self.detailed_evaluation(best_model)
        
        # Make final predictions
        if hasattr(self, 'ensemble_models') and len(self.ensemble_models) > 1:
            print("\nUsing ensemble for final predictions...")
            final_predictions = self.make_ensemble_predictions()
            filename = 'enhanced_ensemble_submission.csv'
        else:
            print("\nUsing single enhanced model for final predictions...")
            test_predictions = best_model.predict(self.X_test, verbose=0)
            final_predictions = np.argmax(test_predictions, axis=1)
            filename = 'enhanced_single_submission.csv'
        
        # Create submission
        submission = self.create_submission(final_predictions, filename)
        
        print("\n" + "="*70)
        print(f"  ENHANCED SOLUTION COMPLETE!")
        print(f"  FINAL ACCURACY: {accuracy:.4f}")
        if accuracy > 0.995:
            print(f"  🎯 TARGET ACHIEVED: >{99.5:.1f}% ✅")
        else:
            print(f"  📊 Close to target, consider more epochs or ensemble")
        print("="*70)
        
        return accuracy

def main():
    """Main function"""
    solver = EnhancedMNISTSolver()
    final_accuracy = solver.run_enhanced_solution()
    return final_accuracy

if __name__ == "__main__":
    main()