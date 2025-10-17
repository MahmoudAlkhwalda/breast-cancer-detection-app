"""
Unified CNN Trainer for Breast Cancer Classification
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import cv2
from .cnn_evaluator import CNNEvaluator

class UnifiedCNNTrainer:
    """Unified CNN trainer for breast cancer classification"""
    
    def __init__(self, input_shape=(224, 224, 3), num_classes=3):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.evaluator = CNNEvaluator(class_names=['benign', 'malignant', 'normal'])
        
    def build_model(self):
        """Build improved CNN model"""
        self.model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Second Conv Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Third Conv Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Fourth Conv Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Dense layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def load_data_from_directory(self, data_dir):
        """Load data from directory structure"""
        images = []
        labels = []
        
        class_names = ['benign', 'malignant', 'normal']
        
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_dir, filename)
                        try:
                            image = cv2.imread(image_path)
                            if image is not None:
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                image = cv2.resize(image, self.input_shape[:2])
                                image = image.astype('float32') / 255.0
                                
                                images.append(image)
                                labels.append(class_idx)
                        except Exception as e:
                            print(f"Error processing image {image_path}: {e}")
                            continue
        
        return np.array(images), np.array(labels)
    
    def prepare_realistic_data_split(self, X, y, test_size=0.2, val_size=0.2):
        """Prepare realistic train/validation/test split with proper stratification"""
        from sklearn.model_selection import train_test_split
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        print(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        print(f"Class distribution - Train: {np.bincount(y_train)}")
        print(f"Class distribution - Val: {np.bincount(y_val)}")
        print(f"Class distribution - Test: {np.bincount(y_test)}")
        print(f"Class weights: {class_weight_dict}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weight_dict
    
    def train(self, train_data, validation_data, epochs=50, batch_size=32, class_weights=None):
        """Train the model with improved parameters and realistic evaluation"""
        if self.model is None:
            self.build_model()
        
        # Enhanced data augmentation with class balancing
        datagen = ImageDataGenerator(
            rotation_range=20,  # Reduced to prevent over-augmentation
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,  # Medical images shouldn't be vertically flipped
            zoom_range=0.15,
            brightness_range=[0.9, 1.1],  # Reduced brightness variation
            shear_range=0.1,
            fill_mode='nearest'
        )
        
        # Callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Monitor loss instead of accuracy
            patience=15,  # Increased patience
            restore_best_weights=True,
            min_delta=0.001
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,  # More aggressive reduction
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model with class weights
        history = self.model.fit(
            datagen.flow(train_data[0], train_data[1], batch_size=batch_size),
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate_model_realistic(self, test_data, save_results=True, output_dir='evaluation_results'):
        """Evaluate model with comprehensive metrics for realistic assessment"""
        print("\n=== COMPREHENSIVE MODEL EVALUATION ===")
        
        # Use the comprehensive evaluator
        evaluation_results = self.evaluator.evaluate_model(
            self.model, 
            test_data, 
            save_results=save_results, 
            output_dir=output_dir
        )
        
        return evaluation_results
    
    def evaluate_model_simple(self, test_data):
        """Simple evaluation for backward compatibility"""
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        import numpy as np
        
        X_test, y_test = test_data
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate F1 scores
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        # Class names
        class_names = ['benign', 'malignant', 'normal']
        
        print(f"\n=== SIMPLE MODEL EVALUATION ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        
        print(f"\n=== CLASSIFICATION REPORT ===")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        print(f"\n=== CONFUSION MATRIX ===")
        cm = confusion_matrix(y_test, y_pred)
        print("Predicted:    Benign  Malignant  Normal")
        for i, class_name in enumerate(class_names):
            print(f"Actual {class_name:8}: {cm[i][0]:7} {cm[i][1]:10} {cm[i][2]:7}")
        
        # Calculate per-class accuracy
        print(f"\n=== PER-CLASS ACCURACY ===")
        for i, class_name in enumerate(class_names):
            class_mask = y_test == i
            class_accuracy = np.sum((y_pred[class_mask] == y_test[class_mask])) / np.sum(class_mask)
            print(f"{class_name}: {class_accuracy:.4f}")
        
        return {
            'accuracy': test_accuracy,
            'loss': test_loss,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        }
    
    def save_model(self, filepath):
        """Save the model"""
        if self.model is not None:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return self.model
    
    def train_realistic_model(self, data_dir, epochs=50, batch_size=32, comprehensive_eval=True, output_dir='evaluation_results'):
        """Complete realistic training pipeline with proper evaluation"""
        print("=== STARTING REALISTIC CNN TRAINING ===")
        
        # Load data
        X, y = self.load_data_from_directory(data_dir)
        print(f"Loaded {len(X)} images with shape {X.shape}")
        
        # Prepare realistic data split
        train_data, val_data, test_data, class_weights = self.prepare_realistic_data_split(X, y)
        
        # Build model
        self.build_model()
        
        # Train model
        print("\n=== TRAINING MODEL ===")
        history = self.train(train_data, val_data, epochs, batch_size, class_weights)
        
        # Evaluate model with comprehensive metrics
        print("\n=== EVALUATING MODEL ===")
        if comprehensive_eval:
            evaluation_results = self.evaluate_model_realistic(test_data, save_results=True, output_dir=output_dir)
        else:
            evaluation_results = self.evaluate_model_simple(test_data)
        
        return history, evaluation_results

if __name__ == '__main__':
    # Example usage
    trainer = UnifiedCNNTrainer()
    model = trainer.build_model()
    print("Unified CNN trainer built successfully!")
    print(model.summary())
