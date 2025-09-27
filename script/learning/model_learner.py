"""
Model Learning System for Breast Cancer Analysis
Allows models to learn from database results and user feedback
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class ModelLearner:
    def __init__(self, db_session=None):
        """
        Initialize the model learning system
        """
        self.db_session = db_session
        self.learning_threshold = 50  # Minimum samples needed for learning
        self.retrain_threshold = 100  # Minimum samples for full retraining
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 20
        
    def get_learning_data(self, data_type='form', verified_only=True):
        """
        Get data from database for learning
        """
        try:
            if not self.db_session:
                print("No database session available")
                return None, None
            
            # Import here to avoid circular imports
            from app import Prediction, ModelTrainingData
            
            if verified_only:
                # Get verified predictions for learning
                predictions = self.db_session.query(Prediction).filter(
                    Prediction.is_verified == True,
                    Prediction.prediction_type == data_type
                ).all()
            else:
                # Get all predictions for learning
                predictions = self.db_session.query(Prediction).filter(
                    Prediction.prediction_type == data_type
                ).all()
            
            print(f"Found {len(predictions)} {data_type} predictions for learning")
            
            if data_type == 'form':
                return self._prepare_form_learning_data(predictions)
            else:
                return self._prepare_image_learning_data(predictions)
                
        except Exception as e:
            print(f"Error getting learning data: {e}")
            return None, None
    
    def _prepare_form_learning_data(self, predictions):
        """
        Prepare form data for learning
        """
        try:
            features_list = []
            labels = []
            
            for prediction in predictions:
                if prediction.form_data and prediction.verified_result:
                    # Extract features from form data
                    features = []
                    for key, value in prediction.form_data.items():
                        if key != 'csrf_token' and key != 'user_notes':
                            try:
                                features.append(float(value))
                            except (ValueError, TypeError):
                                features.append(0.0)
                    
                    if len(features) == 30:  # Ensure we have all 30 features
                        features_list.append(features)
                        
                        # Convert verified result to numeric
                        if prediction.verified_result.lower() == 'malignant':
                            labels.append(0)
                        elif prediction.verified_result.lower() == 'benign':
                            labels.append(1)
                        elif prediction.verified_result.lower() == 'normal':
                            labels.append(2)
            
            if len(features_list) > 0:
                X = np.array(features_list)
                y = np.array(labels)
                print(f"Prepared form learning data: {X.shape[0]} samples")
                return X, y
            else:
                print("No valid form data found for learning")
                return None, None
                
        except Exception as e:
            print(f"Error preparing form learning data: {e}")
            return None, None
    
    def _prepare_image_learning_data(self, predictions):
        """
        Prepare image data for learning
        """
        try:
            images = []
            labels = []
            
            for prediction in predictions:
                if prediction.image_path and prediction.verified_result and os.path.exists(prediction.image_path):
                    try:
                        # Load and preprocess image
                        image = cv2.imread(prediction.image_path)
                        if image is not None:
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, (128, 128))
                            image = image.astype('float32') / 255.0
                            
                            images.append(image)
                            
                            # Convert verified result to numeric
                            if prediction.verified_result.lower() == 'benign':
                                labels.append(0)
                            elif prediction.verified_result.lower() == 'malignant':
                                labels.append(1)
                            elif prediction.verified_result.lower() == 'normal':
                                labels.append(2)
                    except Exception as e:
                        print(f"Error processing image {prediction.image_path}: {e}")
                        continue
            
            if len(images) > 0:
                X = np.array(images)
                y = np.array(labels)
                print(f"Prepared image learning data: {X.shape[0]} samples")
                return X, y
            else:
                print("No valid image data found for learning")
                return None, None
                
        except Exception as e:
            print(f"Error preparing image learning data: {e}")
            return None, None
    
    def learn_from_form_data(self, X, y, model_path=None):
        """
        Learn from form data and update the model
        """
        try:
            if len(X) < self.learning_threshold:
                print(f"Not enough data for learning: {len(X)} < {self.learning_threshold}")
                return False
            
            print(f"Learning from {len(X)} form samples...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train new model
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=15,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Form model learning accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred))
            
            # Save updated model
            if model_path is None:
                model_path = f"models/ml_model_learned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            
            os.makedirs('models', exist_ok=True)
            
            model_data = {
                "model": model,
                "scaler": scaler,
                "feature_names": [f"feature_{i}" for i in range(30)],  # Standard feature names
                "learning_date": datetime.now().isoformat(),
                "accuracy": accuracy,
                "sample_count": len(X)
            }
            
            joblib.dump(model_data, model_path)
            print(f"Learned form model saved to {model_path}")
            
            return True, model_path, accuracy
            
        except Exception as e:
            print(f"Error learning from form data: {e}")
            return False, None, 0.0
    
    def learn_from_image_data(self, X, y, model_path=None):
        """
        Learn from image data and update the CNN model
        """
        try:
            if len(X) < self.learning_threshold:
                print(f"Not enough data for learning: {len(X)} < {self.learning_threshold}")
                return False
            
            print(f"Learning from {len(X)} image samples...")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build CNN model
            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Conv2D(256, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(3, activation='softmax')  # 3 classes: benign, malignant, normal
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Data augmentation
            from tensorflow.keras.preprocessing.image import ImageDataGenerator
            datagen = ImageDataGenerator(
                rotation_range=30,
                width_shift_range=0.3,
                height_shift_range=0.3,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.2,
                brightness_range=[0.8, 1.2]
            )
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001
            )
            
            # Train model
            history = model.fit(
                datagen.flow(X_train, y_train, batch_size=self.batch_size),
                epochs=self.epochs,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            print(f"CNN model learning accuracy: {test_accuracy:.4f}")
            
            # Save updated model
            if model_path is None:
                model_path = f"models/cnn_model_learned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
            
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
            print(f"Learned CNN model saved to {model_path}")
            
            return True, model_path, test_accuracy
            
        except Exception as e:
            print(f"Error learning from image data: {e}")
            return False, None, 0.0
    
    def incremental_learning(self, data_type='form'):
        """
        Perform incremental learning from new data
        """
        try:
            print(f"Starting incremental learning for {data_type} data...")
            
            # Get learning data
            X, y = self.get_learning_data(data_type, verified_only=True)
            
            if X is None or len(X) < self.learning_threshold:
                print(f"Not enough verified data for learning: {len(X) if X is not None else 0}")
                return False
            
            # Learn from data
            if data_type == 'form':
                success, model_path, accuracy = self.learn_from_form_data(X, y)
            else:
                success, model_path, accuracy = self.learn_from_image_data(X, y)
            
            if success:
                print(f"Incremental learning completed successfully!")
                print(f"Model accuracy: {accuracy:.4f}")
                print(f"Model saved to: {model_path}")
                
                # Update model version in database
                self._update_model_version(data_type, model_path, accuracy, len(X))
                
                return True
            else:
                print("Incremental learning failed")
                return False
                
        except Exception as e:
            print(f"Error in incremental learning: {e}")
            return False
    
    def _update_model_version(self, model_type, model_path, accuracy, sample_count):
        """
        Update model version in database
        """
        try:
            from app import ModelVersion
            
            # Deactivate old models
            self.db_session.query(ModelVersion).filter_by(
                model_name=f"{model_type}_model", 
                is_active=True
            ).update({'is_active': False})
            
            # Create new model version
            new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            new_model = ModelVersion(
                model_name=f"{model_type}_model",
                version=new_version,
                file_path=model_path,
                accuracy=accuracy,
                training_data_count=sample_count,
                is_active=True,
                is_production=True
            )
            
            self.db_session.add(new_model)
            self.db_session.commit()
            
            print(f"Model version updated: {new_version}")
            
        except Exception as e:
            print(f"Error updating model version: {e}")
    
    def get_learning_stats(self):
        """
        Get learning statistics
        """
        try:
            from app import Prediction, ModelVersion
            
            stats = {
                'total_predictions': self.db_session.query(Prediction).count(),
                'verified_predictions': self.db_session.query(Prediction).filter_by(is_verified=True).count(),
                'form_predictions': self.db_session.query(Prediction).filter_by(prediction_type='form').count(),
                'image_predictions': self.db_session.query(Prediction).filter_by(prediction_type='image').count(),
                'verified_form_predictions': self.db_session.query(Prediction).filter_by(
                    prediction_type='form', is_verified=True
                ).count(),
                'verified_image_predictions': self.db_session.query(Prediction).filter_by(
                    prediction_type='image', is_verified=True
                ).count(),
                'model_versions': self.db_session.query(ModelVersion).count(),
                'active_models': self.db_session.query(ModelVersion).filter_by(is_active=True).count()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting learning stats: {e}")
            return {}
    
    def should_learn(self, data_type='form'):
        """
        Check if there's enough new data to trigger learning
        """
        try:
            from app import Prediction
            
            # Count verified predictions since last learning
            verified_count = self.db_session.query(Prediction).filter(
                Prediction.prediction_type == data_type,
                Prediction.is_verified == True
            ).count()
            
            return verified_count >= self.learning_threshold
            
        except Exception as e:
            print(f"Error checking learning threshold: {e}")
            return False

def learn_from_database(db_session, data_type='form'):
    """
    Convenience function to learn from database
    """
    learner = ModelLearner(db_session)
    return learner.incremental_learning(data_type)

if __name__ == "__main__":
    # Test the learning system
    print("Model Learning System Test")
    print("=" * 40)
    
    # This would be used with a proper database session
    # learner = ModelLearner()
    # learner.incremental_learning('form')
    # learner.incremental_learning('image')
    
    print("Learning system initialized successfully!")
