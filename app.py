from flask import Flask, render_template, request, jsonify, url_for, flash, redirect, Blueprint, send_from_directory, send_file
from flask_login import LoginManager, login_required, current_user, login_user, logout_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf.csrf import CSRFProtect
import numpy as np
import os
import re
import json
from datetime import datetime
from werkzeug.utils import secure_filename
from config import config
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# Database Models (merged from database.py)
class User(UserMixin, db.Model):
    """User model for authentication and user management"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True)
    uploaded_images = db.relationship('UploadedImage', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Prediction(db.Model):
    """Model to store prediction results"""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    prediction_type = db.Column(db.String(20), nullable=False)  # 'form' or 'image'
    prediction_result = db.Column(db.String(20), nullable=False)  # 'benign', 'malignant', 'normal'
    confidence = db.Column(db.Float, nullable=False)
    
    # For form predictions
    form_data = db.Column(db.JSON, nullable=True)  # Store form features as JSON
    
    # For image predictions
    image_path = db.Column(db.String(255), nullable=True)
    image_filename = db.Column(db.String(255), nullable=True)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    
    # For model improvement
    is_verified = db.Column(db.Boolean, default=False)  # Doctor verification
    verified_result = db.Column(db.String(20), nullable=True)  # Actual diagnosis
    verified_by = db.Column(db.String(100), nullable=True)  # Doctor name
    verified_at = db.Column(db.DateTime, nullable=True)
    
    def __repr__(self):
        return f'<Prediction {self.id}: {self.prediction_result} ({self.confidence:.2f})>'

class UploadedImage(db.Model):
    """Model to store uploaded images and their metadata"""
    __tablename__ = 'uploaded_images'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    file_size = db.Column(db.Integer, nullable=False)
    mime_type = db.Column(db.String(100), nullable=False)
    
    # Image metadata
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    
    # Processing info
    processed = db.Column(db.Boolean, default=False)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=True)
    
    # Timestamps
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<UploadedImage {self.filename}>'

class ModelTrainingData(db.Model):
    """Model to store data for model retraining"""
    __tablename__ = 'model_training_data'
    
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey('predictions.id'), nullable=False)
    
    # Training data type
    data_type = db.Column(db.String(20), nullable=False)  # 'form' or 'image'
    
    # For form data
    features_json = db.Column(db.JSON, nullable=True)
    
    # For image data
    image_path = db.Column(db.String(500), nullable=True)
    image_features = db.Column(db.JSON, nullable=True)  # Extracted features
    
    # Ground truth
    actual_diagnosis = db.Column(db.String(20), nullable=False)
    verified_by_doctor = db.Column(db.Boolean, default=False)
    doctor_name = db.Column(db.String(100), nullable=True)
    
    # Quality metrics
    data_quality_score = db.Column(db.Float, nullable=True)
    is_used_for_training = db.Column(db.Boolean, default=False)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<ModelTrainingData {self.id}: {self.data_type}>'

class ModelVersion(db.Model):
    """Model to track different versions of ML models"""
    __tablename__ = 'model_versions'
    
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)  # 'ml_model' or 'cnn_model'
    version = db.Column(db.String(20), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    
    # Model metrics
    accuracy = db.Column(db.Float, nullable=True)
    precision = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    
    # Training info
    training_data_count = db.Column(db.Integer, nullable=True)
    training_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Status
    is_active = db.Column(db.Boolean, default=False)
    is_production = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<ModelVersion {self.model_name} v{self.version}>'

class SystemLog(db.Model):
    """Model to store system logs and analytics"""
    __tablename__ = 'system_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    log_type = db.Column(db.String(50), nullable=False)  # 'prediction', 'error', 'login', etc.
    message = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    
    # Additional data
    extra_data = db.Column(db.JSON, nullable=True)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SystemLog {self.log_type}: {self.message[:50]}...>'

def init_db(app):
    """Initialize database with app context"""
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create admin user if not exists
        admin_user = User.query.filter_by(username='admin').first()
        if not admin_user:
            from werkzeug.security import generate_password_hash
            admin_user = User(
                username='admin',
                email='admin@breastcancer.app',
                password_hash=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin_user)
            db.session.commit()
            print("Admin user created: admin/admin123")
        
        print("Database initialized successfully!")

def get_db_stats():
    """Get database statistics"""
    stats = {
        'total_users': User.query.count(),
        'total_predictions': Prediction.query.count(),
        'total_images': UploadedImage.query.count(),
        'verified_predictions': Prediction.query.filter_by(is_verified=True).count(),
        'training_data_count': ModelTrainingData.query.count(),
        'model_versions': ModelVersion.query.count()
    }
    return stats

app = Flask(__name__)

# Load configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[config_name])

# Initialize extensions
db.init_app(app)
# csrf = CSRFProtect(app)  # Temporarily disabled
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

# Data management functions (merged from data_manager.py)
def save_form_prediction(user_id, form_data, prediction_result, confidence, ip_address=None, user_agent=None):
    """Save form-based prediction to database"""
    try:
        # Convert form data to proper format
        features = []
        for key, value in form_data.items():
            if key != 'csrf_token':  # Skip CSRF token
                try:
                    features.append(float(value))
                except (ValueError, TypeError):
                    features.append(0.0)
        
        prediction = Prediction(
            user_id=user_id,
            prediction_type='form',
            prediction_result=prediction_result,
            confidence=confidence,
            form_data=form_data,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.session.add(prediction)
        db.session.commit()
        
        # Log the activity
        log_activity('prediction_saved', f'Form prediction saved: {prediction_result}', user_id)
        
        return prediction
        
    except Exception as e:
        db.session.rollback()
        log_activity('prediction_error', f'Failed to save form prediction: {str(e)}', user_id)
        raise e

def _process_uploaded_image(filepath, filename):
    """Process and optimize uploaded image"""
    try:
        from PIL import Image, ImageOps
        import os
        
        # Open and process image
        with Image.open(filepath) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Auto-orient image based on EXIF data
            img = ImageOps.exif_transpose(img)
            
            # Create optimized version
            optimized_path = filepath.replace('.', '_optimized.')
            
            # Save with optimization
            img.save(optimized_path, 
                    format='JPEG', 
                    quality=app.config.get('IMAGE_QUALITY', 95),
                    optimize=True)
            
            # Create thumbnail
            thumbnail_path = filepath.replace('.', '_thumb.')
            thumbnail = img.copy()
            thumbnail.thumbnail(app.config.get('THUMBNAIL_SIZE', (300, 300)), Image.Resampling.LANCZOS)
            thumbnail.save(thumbnail_path, format='JPEG', quality=85, optimize=True)
            
            # Create preview
            preview_path = filepath.replace('.', '_preview.')
            preview = img.copy()
            preview.thumbnail(app.config.get('PREVIEW_SIZE', (600, 600)), Image.Resampling.LANCZOS)
            preview.save(preview_path, format='JPEG', quality=90, optimize=True)
            
            # Remove original if different from optimized
            if optimized_path != filepath:
                os.remove(filepath)
                return optimized_path
            
            return filepath
            
    except Exception as e:
        print(f"Error processing image: {e}")
        return filepath

def save_image_prediction(user_id, image_file, prediction_result, confidence, ip_address=None, user_agent=None, filepath=None):
    """Save image-based prediction to database"""
    try:
        # Use provided filepath or save uploaded image
        if filepath is None:
            filename = secure_filename(image_file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            image_file.save(filepath)
        else:
            filename = os.path.basename(filepath)
        
        # Process and optimize image
        processed_filepath = _process_uploaded_image(filepath, filename)
        
        # Get image metadata
        from PIL import Image
        with Image.open(processed_filepath) as img:
            width, height = img.size
        
        # Save image record
        uploaded_image = UploadedImage(
            user_id=user_id,
            filename=filename,
            original_filename=image_file.filename,
            file_path=processed_filepath,
            file_size=os.path.getsize(processed_filepath),
            mime_type=image_file.content_type,
            width=width,
            height=height,
            processed=True
        )
        
        db.session.add(uploaded_image)
        db.session.flush()  # Get the ID
        
        # Save prediction record
        prediction = Prediction(
            user_id=user_id,
            prediction_type='image',
            prediction_result=prediction_result,
            confidence=confidence,
            image_path=processed_filepath,
            image_filename=filename,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        db.session.add(prediction)
        db.session.flush()  # Get the ID
        
        # Link image to prediction
        uploaded_image.prediction_id = prediction.id
        db.session.commit()
        
        log_activity('prediction_saved', f'Image prediction saved: {prediction_result}', user_id)
        
        return prediction, uploaded_image
        
    except Exception as e:
        db.session.rollback()
        log_activity('prediction_error', f'Failed to save image prediction: {str(e)}', user_id)
        raise e

def verify_prediction(prediction_id, verified_result, doctor_name):
    """Mark prediction as verified by doctor"""
    try:
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            raise ValueError("Prediction not found")
        
        prediction.is_verified = True
        prediction.verified_result = verified_result
        prediction.verified_by = doctor_name
        prediction.verified_at = datetime.utcnow()
        
        db.session.commit()
        
        # Create training data entry
        create_training_data(prediction, verified_result)
        
        log_activity('prediction_verified', f'Prediction {prediction_id} verified as {verified_result}', prediction.user_id)
        
        return True
        
    except Exception as e:
        db.session.rollback()
        log_activity('verification_error', f'Failed to verify prediction: {str(e)}')
        raise e

def create_training_data(prediction, verified_result):
    """Create training data entry for model improvement"""
    try:
        training_data = ModelTrainingData(
            prediction_id=prediction.id,
            data_type=prediction.prediction_type,
            actual_diagnosis=verified_result,
            verified_by_doctor=True,
            doctor_name=prediction.verified_by
        )
        
        if prediction.prediction_type == 'form':
            training_data.features_json = prediction.form_data
        else:
            training_data.image_path = prediction.image_path
            # Extract image features for CNN training
            image_features = extract_image_features(prediction.image_path)
            training_data.image_features = image_features
        
        db.session.add(training_data)
        db.session.commit()
        
    except Exception as e:
        log_activity('training_data_error', f'Failed to create training data: {str(e)}')

def extract_image_features(image_path):
    """Extract features from image for training"""
    try:
        import cv2
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to standard size
        image = cv2.resize(image, (128, 128))
        
        # Extract basic features
        features = {
            'mean_rgb': np.mean(image, axis=(0, 1)).tolist(),
            'std_rgb': np.std(image, axis=(0, 1)).tolist(),
            'histogram': cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]).flatten().tolist()
        }
        
        return features
        
    except Exception as e:
        log_activity('feature_extraction_error', f'Failed to extract image features: {str(e)}')
        return None

def get_user_predictions(user_id, limit=50):
    """Get predictions for a specific user"""
    return Prediction.query.filter_by(user_id=user_id)\
                          .order_by(Prediction.created_at.desc())\
                          .limit(limit).all()

def get_prediction_stats():
    """Get prediction statistics"""
    stats = {
        'total_predictions': Prediction.query.count(),
        'form_predictions': Prediction.query.filter_by(prediction_type='form').count(),
        'image_predictions': Prediction.query.filter_by(prediction_type='image').count(),
        'verified_predictions': Prediction.query.filter_by(is_verified=True).count(),
        'benign_predictions': Prediction.query.filter_by(prediction_result='benign').count(),
        'malignant_predictions': Prediction.query.filter_by(prediction_result='malignant').count(),
        'normal_predictions': Prediction.query.filter_by(prediction_result='normal').count()
    }
    return stats

def export_training_data(output_path):
    """Export training data for model retraining"""
    try:
        training_data = ModelTrainingData.query.filter_by(verified_by_doctor=True).all()
        
        export_data = {
            'form_data': [],
            'image_data': [],
            'metadata': {
                'export_date': datetime.utcnow().isoformat(),
                'total_records': len(training_data),
                'verified_records': len([d for d in training_data if d.verified_by_doctor])
            }
        }
        
        for data in training_data:
            if data.data_type == 'form':
                export_data['form_data'].append({
                    'features': data.features_json,
                    'diagnosis': data.actual_diagnosis,
                    'verified': data.verified_by_doctor
                })
            else:
                export_data['image_data'].append({
                    'image_path': data.image_path,
                    'features': data.image_features,
                    'diagnosis': data.actual_diagnosis,
                    'verified': data.verified_by_doctor
                })
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        log_activity('data_exported', f'Training data exported to {output_path}')
        return True
        
    except Exception as e:
        log_activity('export_error', f'Failed to export training data: {str(e)}')
        return False

# Model retraining functions (merged from model_retrainer.py)
def check_retraining_needed():
    """Check if model needs retraining based on new data"""
    # Count new verified data since last training
    last_training = ModelVersion.query.filter_by(
        model_name='ml_model', 
        is_active=True
    ).first()
    
    if not last_training:
        return True, "No previous model found"
    
    new_data_count = ModelTrainingData.query.filter(
        ModelTrainingData.created_at > last_training.training_date,
        ModelTrainingData.verified_by_doctor == True
    ).count()
    
    # Retrain if we have at least 20 new verified samples (reduced threshold)
    if new_data_count >= 20:
        return True, f"Found {new_data_count} new verified samples"
    
    return False, f"Only {new_data_count} new samples available (need 20+)"

def prepare_ml_training_data():
    """Prepare data for ML model retraining - combines original data with new user data"""
    try:
        # Load original breast cancer dataset
        from sklearn.datasets import load_breast_cancer
        from sklearn.preprocessing import StandardScaler
        
        # Get original dataset
        original_data = load_breast_cancer()
        X_original = original_data.data
        y_original = original_data.target
        
        # Get new user data
        training_data = ModelTrainingData.query.filter_by(
            data_type='form',
            verified_by_doctor=True
        ).all()
        
        print(f"Original dataset: {len(X_original)} samples")
        print(f"New user data: {len(training_data)} samples")
        
        if len(training_data) > 0:
            # Prepare new user data
            new_features_list = []
            new_labels = []
            
            for data in training_data:
                if data.features_json:
                    # Convert features to array
                    features = []
                    for key, value in data.features_json.items():
                        if key != 'csrf_token':
                            try:
                                features.append(float(value))
                            except (ValueError, TypeError):
                                features.append(0.0)
                    
                    new_features_list.append(features)
                    new_labels.append(data.actual_diagnosis)
            
            if len(new_features_list) > 0:
                # Convert new data to numpy arrays
                X_new = np.array(new_features_list)
                y_new = np.array(new_labels)
                
                # Combine original and new data
                X_combined = np.vstack([X_original, X_new])
                y_combined = np.hstack([y_original, y_new])
                
                print(f"Combined dataset: {len(X_combined)} samples")
                return X_combined, y_combined
        
        # If no new data, use original dataset
        print("Using original dataset only")
        return X_original, y_original
        
    except Exception as e:
        print(f"Error preparing ML training data: {e}")
        # Fallback to original dataset
        from sklearn.datasets import load_breast_cancer
        original_data = load_breast_cancer()
        return original_data.data, original_data.target

def prepare_cnn_training_data():
    """Prepare data for CNN model retraining - combines original data with new user data"""
    try:
        import cv2
        import os
        
        # Load original dataset from data folder
        original_images = []
        original_labels = []
        
        # Load original training data
        data_dir = 'data/train'
        if os.path.exists(data_dir):
            for class_name in ['benign', 'malignant', 'normal']:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.exists(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(class_dir, filename)
                            try:
                                image = cv2.imread(image_path)
                                if image is not None:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    image = cv2.resize(image, app.config['IMAGE_SIZE'])
                                    image = image.astype('float32') / 255.0
                                    
                                    original_images.append(image)
                                    
                                    # Convert class name to numeric
                                    if class_name == 'benign':
                                        original_labels.append(0)
                                    elif class_name == 'malignant':
                                        original_labels.append(1)
                                    else:  # normal
                                        original_labels.append(2)
                            except Exception as e:
                                print(f"Error processing original image {image_path}: {e}")
                                continue
        
        print(f"Original images: {len(original_images)} samples")
        
        # Get new user data
        training_data = ModelTrainingData.query.filter_by(
            data_type='image',
            verified_by_doctor=True
        ).all()
        
        new_images = []
        new_labels = []
        
        for data in training_data:
            if data.image_path and os.path.exists(data.image_path):
                try:
                    # Load and preprocess image
                    image = cv2.imread(data.image_path)
                    if image is not None:
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(image, app.config['IMAGE_SIZE'])
                        image = image.astype('float32') / 255.0
                        
                        new_images.append(image)
                        
                        # Convert diagnosis to numeric
                        if data.actual_diagnosis == 'benign':
                            new_labels.append(0)
                        elif data.actual_diagnosis == 'malignant':
                            new_labels.append(1)
                        else:  # normal
                            new_labels.append(2)
                            
                except Exception as e:
                    print(f"Error processing new image {data.image_path}: {e}")
                    continue
        
        print(f"New user images: {len(new_images)} samples")
        
        # Combine original and new data
        if len(new_images) > 0:
            all_images = original_images + new_images
            all_labels = original_labels + new_labels
            
            print(f"Combined dataset: {len(all_images)} samples")
            return np.array(all_images), np.array(all_labels)
        else:
            print("Using original dataset only")
            return np.array(original_images), np.array(original_labels)
            
    except Exception as e:
        print(f"Error preparing CNN training data: {e}")
        # Fallback to original dataset only
        import cv2
        import os
        
        images = []
        labels = []
        
        data_dir = 'data/train'
        if os.path.exists(data_dir):
            for class_name in ['benign', 'malignant', 'normal']:
                class_dir = os.path.join(data_dir, class_name)
                if os.path.exists(class_dir):
                    for filename in os.listdir(class_dir):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                            image_path = os.path.join(class_dir, filename)
                            try:
                                image = cv2.imread(image_path)
                                if image is not None:
                                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                    image = cv2.resize(image, app.config['IMAGE_SIZE'])
                                    image = image.astype('float32') / 255.0
                                    
                                    images.append(image)
                                    
                                    if class_name == 'benign':
                                        labels.append(0)
                                    elif class_name == 'malignant':
                                        labels.append(1)
                                    else:  # normal
                                        labels.append(2)
                            except Exception as e:
                                continue
        
        return np.array(images), np.array(labels)

def retrain_ml_model():
    """Retrain the ML model with combined original and new data"""
    try:
        print("Starting ML model retraining with combined data...")
        
        # Prepare combined data
        X, y = prepare_ml_training_data()
        
        print(f"Training with {len(X)} samples")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train new model with improved parameters
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=200,  # Increased for better performance
            random_state=42,
            max_depth=15,      # Increased depth
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        from sklearn.metrics import accuracy_score, classification_report
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"ML Model accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Save new model and scaler
        new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_model_path = f"models/ml_model_{new_version}.pkl"
        scaler_path = f"models/scaler_{new_version}.pkl"
        
        os.makedirs('models', exist_ok=True)
        import joblib
        joblib.dump(model, new_model_path)
        joblib.dump(scaler, scaler_path)
        
        # Update database
        update_model_version('ml_model', new_version, new_model_path, accuracy)
        
        # Mark training data as used
        mark_data_as_used('form')
        
        print(f"ML model retrained successfully! New version: {new_version}")
        return True, f"ML model retrained with accuracy: {accuracy:.4f}"
        
    except Exception as e:
        print(f"Error retraining ML model: {e}")
        return False, str(e)

def retrain_cnn_model():
    """Retrain the CNN model with combined original and new data"""
    try:
        print("Starting CNN model retraining with combined data...")
        
        # Prepare combined data
        X, y = prepare_cnn_training_data()
        
        print(f"Training with {len(X)} images")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build improved CNN model
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        model = Sequential([
            # First Conv Block
            Conv2D(32, (3, 3), activation='relu', input_shape=(*app.config['IMAGE_SIZE'], 3)),
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
            Dense(3, activation='softmax')  # 3 classes: benign, malignant, normal
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Enhanced data augmentation
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
        
        # Callbacks for better training
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
            datagen.flow(X_train, y_train, batch_size=32),
            epochs=50,  # Increased epochs
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        
        print(f"CNN Model accuracy: {test_accuracy:.4f}")
        
        # Save new model
        new_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        new_model_path = f"models/cnn_model_{new_version}.h5"
        
        os.makedirs('models', exist_ok=True)
        model.save(new_model_path)
        
        # Update database
        update_model_version('cnn_model', new_version, new_model_path, test_accuracy)
        
        # Mark training data as used
        mark_data_as_used('image')
        
        print(f"CNN model retrained successfully! New version: {new_version}")
        return True, f"CNN model retrained with accuracy: {test_accuracy:.4f}"
        
    except Exception as e:
        print(f"Error retraining CNN model: {e}")
        return False, str(e)

def update_model_version(model_name, version, file_path, accuracy):
    """Update model version in database"""
    # Deactivate old models
    ModelVersion.query.filter_by(model_name=model_name, is_active=True).update({'is_active': False})
    
    # Create new model version
    new_model = ModelVersion(
        model_name=model_name,
        version=version,
        file_path=file_path,
        accuracy=accuracy,
        training_data_count=ModelTrainingData.query.filter_by(
            data_type='form' if model_name == 'ml_model' else 'image',
            verified_by_doctor=True
        ).count(),
        is_active=True,
        is_production=True
    )
    
    db.session.add(new_model)
    db.session.commit()

def mark_data_as_used(data_type):
    """Mark training data as used for training"""
    ModelTrainingData.query.filter_by(
        data_type=data_type,
        verified_by_doctor=True,
        is_used_for_training=False
    ).update({'is_used_for_training': True})
    
    db.session.commit()

def auto_retrain_if_needed():
    """Automatically retrain models if enough new data is available"""
    try:
        # Check if retraining is needed
        ml_needed, ml_reason = check_retraining_needed()
        
        results = []
        
        if ml_needed:
            print(f"ML retraining needed: {ml_reason}")
            success, message = retrain_ml_model()
            results.append(f"ML Model: {message}")
        
        # Check CNN separately with reduced threshold for combined training
        cnn_data_count = ModelTrainingData.query.filter_by(
            data_type='image',
            verified_by_doctor=True,
            is_used_for_training=False
        ).count()
        
        if cnn_data_count >= 20:  # Reduced threshold for combined training
            print(f"CNN retraining needed: {cnn_data_count} new image samples")
            success, message = retrain_cnn_model()
            results.append(f"CNN Model: {message}")
        
        if not results:
            return "No retraining needed at this time"
        
        return " | ".join(results)
        
    except Exception as e:
        return f"Error in auto retraining: {str(e)}"

def get_training_stats():
    """Get training statistics"""
    stats = {
        'total_verified_data': ModelTrainingData.query.filter_by(verified_by_doctor=True).count(),
        'form_data': ModelTrainingData.query.filter_by(data_type='form', verified_by_doctor=True).count(),
        'image_data': ModelTrainingData.query.filter_by(data_type='image', verified_by_doctor=True).count(),
        'unused_data': ModelTrainingData.query.filter_by(is_used_for_training=False, verified_by_doctor=True).count(),
        'ml_models': ModelVersion.query.filter_by(model_name='ml_model').count(),
        'cnn_models': ModelVersion.query.filter_by(model_name='cnn_model').count(),
        'active_ml_model': ModelVersion.query.filter_by(model_name='ml_model', is_active=True).first(),
        'active_cnn_model': ModelVersion.query.filter_by(model_name='cnn_model', is_active=True).first()
    }
    return stats

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Authentication helper functions
def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"

def log_activity(log_type, message, user_id=None, metadata=None):
    """Log system activity"""
    log = SystemLog(
        log_type=log_type,
        message=message,
        user_id=user_id,
        ip_address=request.remote_addr,
        user_agent=request.headers.get('User-Agent'),
        extra_data=metadata
    )
    db.session.add(log)
    db.session.commit()

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from script.ml_models.model import feature_names, predict_from_form
from script.cnn.unified_cnn_predictor import get_cnn_prediction, initialize_cnn_predictor
from script.learning.model_learner import ModelLearner, learn_from_database
from script.learning.feedback_system import FeedbackSystem, submit_prediction_feedback
from script.learning.learning_scheduler import initialize_learning_scheduler, get_learning_scheduler
from script.learning.automatic_learner import AutomaticLearner, process_prediction_for_learning, trigger_automatic_learning

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember')
        
        if not username or not password:
            flash('Please fill in all fields', 'error')
            return render_template('auth/login.html')
        
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=bool(remember))
            user.last_login = datetime.utcnow()
            db.session.commit()
            
            log_activity('login', f'User {username} logged in', user.id)
            flash('Login successful!', 'success')
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            else:
                # Redirect to dashboard if user is admin, otherwise to index
                if user.is_admin:
                    return redirect(url_for('admin_dashboard'))
                else:
                    return redirect(url_for('dashboard'))
        else:
            log_activity('login_failed', f'Failed login attempt for username: {username}')
            flash('Invalid username or password', 'error')
    
    return render_template('auth/login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        print(f"🔍 REGISTRATION ATTEMPT:")
        print(f"   Username: {username}")
        print(f"   Email: {email}")
        print(f"   Password length: {len(password) if password else 0}")
        print(f"   Confirm password length: {len(confirm_password) if confirm_password else 0}")
        
        # Validation
        if not all([username, email, password, confirm_password]):
            print("Missing required fields")
            flash('Please fill in all fields', 'error')
            return render_template('auth/register.html')
        
        if password != confirm_password:
            print(" Passwords do not match")
            flash('Passwords do not match', 'error')
            return render_template('auth/register.html')
        
        if not validate_email(email):
            print("Invalid email format")
            flash('Invalid email format', 'error')
            return render_template('auth/register.html')
        
        is_valid, message = validate_password(password)
        if not is_valid:
            print(f"Password validation failed: {message}")
            flash(message, 'error')
            return render_template('auth/register.html')
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            print("Username already exists")
            flash('Username already exists', 'error')
            return render_template('auth/register.html')
        
        if User.query.filter_by(email=email).first():
            print("Email already registered")
            flash('Email already registered', 'error')
            return render_template('auth/register.html')
        
        # Create new user
        try:
            print("All validations passed, creating user...")
            user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            db.session.add(user)
            db.session.commit()
            
            print(f"User '{username}' created successfully!")
            log_activity('user_registered', f'New user registered: {username}', user.id)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except Exception as e:
            print(f" Database error: {e}")
            print(f" Error type: {type(e).__name__}")
            print(f" Error details: {str(e)}")
            db.session.rollback()
            log_activity('registration_error', f'Registration failed for {username}: {str(e)}')
            flash('Registration failed. Please try again.', 'error')
            return render_template('auth/register.html')
    
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    username = current_user.username
    logout_user()
    log_activity('logout', f'User {username} logged out')
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/profile')
@login_required
def profile():
    """User profile page"""
    user_predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).limit(10).all()
    return render_template('auth/profile.html', predictions=user_predictions)

@app.route('/change-password', methods=['GET', 'POST'])
@login_required
def change_password():
    """Change user password"""
    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if not check_password_hash(current_user.password_hash, current_password):
            flash('Current password is incorrect', 'error')
            return render_template('auth/change_password.html')
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return render_template('auth/change_password.html')
        
        is_valid, message = validate_password(new_password)
        if not is_valid:
            flash(message, 'error')
            return render_template('auth/change_password.html')
        
        current_user.password_hash = generate_password_hash(new_password)
        db.session.commit()
        
        log_activity('password_changed', f'User {current_user.username} changed password', current_user.id)
        flash('Password changed successfully!', 'success')
        return redirect(url_for('profile'))
    
    return render_template('auth/change_password.html')

@app.route('/api/user-stats')
@login_required
def user_stats():
    """Get user statistics for dashboard"""
    stats = {
        'total_predictions': len(current_user.predictions),
        'form_predictions': len([p for p in current_user.predictions if p.prediction_type == 'form']),
        'image_predictions': len([p for p in current_user.predictions if p.prediction_type == 'image']),
        'uploaded_images': len(current_user.uploaded_images),
        'last_prediction': current_user.predictions.order_by(Prediction.created_at.desc()).first()
    }
    return jsonify(stats)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form-predict')
def form_predict():
    return render_template('form_predict.html', feature_names=feature_names)

@app.route('/form-predict-simple')
def form_predict_simple():
    return render_template('form_predict_simple.html', feature_names=feature_names)

@app.route('/debug-form')
def debug_form():
    return render_template('debug_form.html')

@app.route('/test-predict', methods=['POST'])
def test_predict():
    """Simple test route to verify form submission works"""
    try:
        print("TEST PREDICTION ROUTE CALLED")
        form_data = request.form.to_dict()
        print(f"Received {len(form_data)} fields")
        print(f"Sample data: {list(form_data.items())[:3]}")
        
        return f"""
        <html>
        <body>
            <h1>Test Prediction Success!</h1>
            <p>Form submission is working correctly.</p>
            <p>Received {len(form_data)} fields</p>
            <p>Sample data: {list(form_data.items())[:3]}</p>
            <a href="/form-predict">Back to Form</a>
        </body>
        </html>
        """
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/form-test')
def form_test():
    return render_template('form_test.html')



@app.route('/image-predict')
def image_predict():
    return render_template('image_predict.html')
##API
@app.route('/predict-form', methods=['POST'])
def predict_form():
    try:
        print("=" * 50)
        print("FORM PREDICTION REQUESTED")
        print("=" * 50)
        print("Form prediction requested")
        print(f"Request method: {request.method}")
        print(f"Request URL: {request.url}")
        print(f"Request headers: {dict(request.headers)}")
        
        form_data = request.form.to_dict()
        print(f"Form data received: {len(form_data)} fields")
        print(f"Form data keys: {list(form_data.keys())[:5]}...")
        print(f"Form data values: {list(form_data.values())[:5]}...")
        print(f"Request method: {request.method}")
        print(f"Content type: {request.content_type}")
        print(f"Request headers: {dict(request.headers)}")
        
        # Validate that we have the required features
        if len(form_data) < len(feature_names):
            print(f"Warning: Only {len(form_data)} fields received, expected {len(feature_names)}")
            flash(f'Warning: Only {len(form_data)} fields received, expected {len(feature_names)}', 'warning')
        
        features = []
        missing_features = []
        zero_count = 0
        
        for feature in feature_names:
            value = form_data.get(feature, '')
            try:
                # Handle empty strings and convert to float
                if value == '' or value is None:
                    value = '0'
                    missing_features.append(feature)
                
                float_value = float(value)
                features.append(float_value)
                
                # Count zero values
                if float_value == 0:
                    zero_count += 1
                    
            except (ValueError, TypeError) as e:
                print(f"Invalid value for {feature}: {value}, using 0")
                features.append(0.0)
                missing_features.append(feature)
                zero_count += 1
        
        # Check if all values are zero (no meaningful input)
        if zero_count == len(feature_names):
            print("Warning: All form values are zero - no meaningful input provided")
            flash('Please enter actual values for the prediction. All fields cannot be zero.', 'error')
            return render_template('form_predict.html', feature_names=feature_names)
        
        if missing_features:
            print(f"Missing or invalid features: {missing_features}")
            flash(f'Some features had invalid values and were set to 0: {", ".join(missing_features[:5])}', 'warning')
        
        # Warn if most values are zero (but allow prediction)
        if zero_count > len(feature_names) * 0.8:  # More than 80% are zero
            print(f"Warning: {zero_count}/{len(feature_names)} values are zero - prediction may be less accurate")
            flash(f'Warning: {zero_count} out of {len(feature_names)} values are zero. Prediction accuracy may be reduced.', 'warning')
        
        print(f"Features array length: {len(features)}")
        print(f"Sample features: {features[:5]}")  
        
        # Check if model is available
        if not hasattr(predict_from_form, '__call__'):
            return render_template('error.html', error_message="ML model not available")
        
        # Validate features array
        if len(features) != len(feature_names):
            return render_template('error.html', error_message=f"Invalid number of features: expected {len(feature_names)}, got {len(features)}")
        
        prediction, confidence = predict_from_form(np.array([features]))
        
        print(f"🔍 FORM PREDICTION: {prediction} (confidence: {confidence:.3f})")
        print(f"📊 Form data processed: {len(features)} features")
        
        # Validate prediction result
        if prediction == "Error":
            return render_template('error.html', error_message="Prediction failed. Please check your input values.")
        
        # Save prediction to database (if user is logged in)
        saved_prediction = None
        if current_user.is_authenticated:
            try:
                saved_prediction = save_form_prediction(
                    user_id=current_user.id,
                    form_data=form_data,
                    prediction_result=prediction,
                    confidence=confidence,
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                
                # Trigger automatic learning for this prediction
                if saved_prediction:
                    try:
                        print(f"🧠 Starting automatic learning for form prediction {saved_prediction.id}...")
                        process_prediction_for_learning(db.session, saved_prediction)
                        print(f"✅ Automatic learning completed for form prediction {saved_prediction.id}")
                    except Exception as e:
                        print(f"❌ Automatic learning failed for form prediction: {e}")
                        # Continue without failing the main prediction
                        
            except Exception as e:
                print(f"Error saving prediction: {e}")
                # Continue without saving to database
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'input_type': 'form',
            'prediction_id': saved_prediction.id if saved_prediction else None
        }
        
        print("=" * 50)
        print("RENDERING RESULTS")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence}")
        print(f"Result dict: {result}")
        print("=" * 50)
        
        # Use the main results template
        try:
            return render_template('results.html', result=result)
        except Exception as template_error:
            print(f"Template rendering error: {template_error}")
            # Fallback: return a simple HTML response
            return f"""
            <html>
            <body>
                <h1>Prediction Results</h1>
                <p><strong>Prediction:</strong> {prediction}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Type:</strong> {result.get('input_type', 'form')}</p>
                <a href="/form-predict">Try Again</a>
            </body>
            </html>
            """
    
    except Exception as e:
        print(f"Error in form prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error_message=f"Prediction failed: {str(e)}")

@app.route('/predict-image', methods=['POST'])
@login_required
def predict_image():
    try:
        print("Image prediction requested")
        
        if 'file' not in request.files:
            return render_template('error.html', error_message='No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', error_message='No file selected')
        
        if not file or not allowed_file(file.filename):
            return render_template('error.html', error_message='Invalid file type. Please upload a PNG, JPG, JPEG, BMP, or TIFF image.')
        
        # Validate file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return render_template('error.html', error_message=f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB')
        
        # Save file temporarily to get path
        temp_filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_filename = f"{timestamp}_{temp_filename}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(temp_filepath), exist_ok=True)
        
        # Save file temporarily
        file.save(temp_filepath)
        
        try:
            # Validate that the file is a valid image
            from PIL import Image
            try:
                with Image.open(temp_filepath) as img:
                    img.verify()  # Verify it's a valid image
                print(f"Image validated: {img.size}")
            except Exception as e:
                return render_template('error.html', error_message=f'Invalid image file: {str(e)}')
            
            # Get prediction
            prediction, confidence, message = get_cnn_prediction(temp_filepath)
            
            print(f"🖼️ IMAGE PREDICTION: {prediction} (confidence: {confidence:.3f})")
            print(f"📊 Image processed: {img.size[0]}x{img.size[1]} pixels")
            print(f"💬 Analysis message: {message}")
            
            # Validate prediction result
            if prediction == "Error":
                return render_template('error.html', error_message="Image analysis failed. Please try with a different image.")
            
            print(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Method: {message}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        
        # Save file permanently for database
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save file permanently
        file.seek(0)  # Reset file pointer
        file.save(filepath)
        
        # Save prediction to database
        try:
            saved_prediction, uploaded_image = save_image_prediction(
                user_id=current_user.id,
                image_file=file,
                prediction_result=prediction,
                confidence=confidence,
                ip_address=request.remote_addr,
                user_agent=request.headers.get('User-Agent'),
                filepath=filepath
            )
            
            # Trigger automatic learning for this prediction
            if saved_prediction:
                try:
                    print(f"🧠 Starting automatic learning for image prediction {saved_prediction.id}...")
                    process_prediction_for_learning(db.session, saved_prediction)
                    print(f"✅ Automatic learning completed for image prediction {saved_prediction.id}")
                except Exception as e:
                    print(f"❌ Automatic learning failed for image prediction: {e}")
                    # Continue without failing the main prediction
                    
        except Exception as e:
            print(f"Error saving prediction to database: {e}")
            # Continue without saving to database
            saved_prediction = None
            uploaded_image = None
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'image_url': url_for('uploaded_file', filename=uploaded_image.filename) if uploaded_image else None,
            'input_type': 'image',
            'prediction_id': saved_prediction.id if saved_prediction else None,
            'method': message
        }
        
        return render_template('results.html', result=result)
    
    except Exception as e:
        print(f"Error in image prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('error.html', error_message=f"Image analysis failed: {str(e)}")

# Dashboard routes
@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard"""
    user_predictions = get_user_predictions(current_user.id, limit=10)
    stats = get_prediction_stats()
    return render_template('dashboard.html', predictions=user_predictions, stats=stats)

@app.route('/my-predictions')
@login_required
def my_predictions():
    """User's prediction history"""
    predictions = get_user_predictions(current_user.id, limit=100)
    return render_template('my_predictions.html', predictions=predictions)

@app.route('/edit-prediction/<int:prediction_id>', methods=['GET', 'POST'])
@login_required
def edit_prediction(prediction_id):
    """Edit a prediction"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if user owns this prediction
    if prediction.user_id != current_user.id:
        flash('Access denied. You can only edit your own predictions.', 'error')
        return redirect(url_for('my_predictions'))
    
    if request.method == 'POST':
        try:
            # Update prediction details
            prediction.prediction_result = request.form.get('prediction_result')
            prediction.confidence = float(request.form.get('confidence', 0))
            
            # Add notes if provided
            notes = request.form.get('notes', '').strip()
            if notes:
                # Store notes in form_data if it's a form prediction, or create a new field
                if prediction.prediction_type == 'form' and prediction.form_data:
                    prediction.form_data['user_notes'] = notes
                else:
                    # For image predictions, store in a new field or form_data
                    if not prediction.form_data:
                        prediction.form_data = {}
                    prediction.form_data['user_notes'] = notes
            
            db.session.commit()
            flash('Prediction updated successfully!', 'success')
            return redirect(url_for('my_predictions'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating prediction: {str(e)}', 'error')
    
    return render_template('edit_prediction.html', prediction=prediction)

@app.route('/delete-prediction/<int:prediction_id>', methods=['POST'])
@login_required
def delete_prediction(prediction_id):
    """Delete a prediction"""
    prediction = Prediction.query.get_or_404(prediction_id)
    
    # Check if user owns this prediction
    if prediction.user_id != current_user.id:
        flash('Access denied. You can only delete your own predictions.', 'error')
        return redirect(url_for('my_predictions'))
    
    try:
        # Delete associated image file if it exists
        if prediction.image_path and os.path.exists(prediction.image_path):
            os.remove(prediction.image_path)
        
        db.session.delete(prediction)
        db.session.commit()
        flash('Prediction deleted successfully!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting prediction: {str(e)}', 'error')
    
    return redirect(url_for('my_predictions'))

@app.route('/bulk-delete-predictions', methods=['POST'])
@login_required
def bulk_delete_predictions():
    """Delete multiple predictions"""
    prediction_ids = request.form.getlist('prediction_ids')
    
    # Debug logging
    print(f"Received prediction_ids: {prediction_ids}")
    print(f"Form data: {request.form}")
    
    if not prediction_ids:
        flash('No predictions selected for deletion.', 'warning')
        return redirect(url_for('my_predictions'))
    
    try:
        deleted_count = 0
        for prediction_id in prediction_ids:
            prediction = Prediction.query.get(prediction_id)
            if prediction and prediction.user_id == current_user.id:
                # Delete associated image file if it exists
                if prediction.image_path and os.path.exists(prediction.image_path):
                    os.remove(prediction.image_path)
                
                db.session.delete(prediction)
                deleted_count += 1
        
        db.session.commit()
        flash(f'Successfully deleted {deleted_count} prediction(s)!', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting predictions: {str(e)}', 'error')
    
    return redirect(url_for('my_predictions'))

# Admin routes
@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin dashboard"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    stats = get_db_stats()
    prediction_stats = get_prediction_stats()
    
    # Get recent predictions for admin
    recent_predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(20).all()
    
    return render_template('admin/dashboard.html', 
                         db_stats=stats, 
                         prediction_stats=prediction_stats,
                         recent_predictions=recent_predictions)

@app.route('/admin/verify-prediction/<int:prediction_id>', methods=['POST'])
@login_required
def admin_verify_prediction(prediction_id):
    """Verify prediction by doctor"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    verified_result = request.form.get('verified_result')
    doctor_name = request.form.get('doctor_name', current_user.username)
    
    try:
        verify_prediction(prediction_id, verified_result, doctor_name)
        return jsonify({'success': True, 'message': 'Prediction verified successfully!'})
    except Exception as e:
        return jsonify({'error': f'Error verifying prediction: {str(e)}'}), 500

@app.route('/admin/export-data')
@login_required
def admin_export_data():
    """Export training data for model improvement"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    try:
        export_path = os.path.join('data', 'training_data_export.json')
        os.makedirs('data', exist_ok=True)
        
        if export_training_data(export_path):
            flash('Training data exported successfully!', 'success')
            return send_file(export_path, as_attachment=True, download_name='training_data_export.json')
        else:
            flash('Failed to export training data', 'error')
    except Exception as e:
        flash(f'Error exporting data: {str(e)}', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/retrain-models')
@login_required
def admin_retrain_models():
    """Manually trigger model retraining"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    try:
        result = auto_retrain_if_needed()
        flash(f'Retraining completed: {result}', 'success')
    except Exception as e:
        flash(f'Error during retraining: {str(e)}', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/retrain-models-api', methods=['POST'])
@login_required
def admin_retrain_models_api():
    """API endpoint for model retraining"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    try:
        result = auto_retrain_if_needed()
        return jsonify({'success': True, 'message': f'Retraining completed: {result}'})
    except Exception as e:
        return jsonify({'error': f'Error during retraining: {str(e)}'}), 500

@app.route('/admin/training-stats')
@login_required
def admin_training_stats():
    """Get training statistics"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    try:
        stats = get_training_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/learning-dashboard')
@login_required
def learning_dashboard():
    """AI Learning Dashboard"""
    if not current_user.is_admin:
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    try:
        learner = ModelLearner(db.session)
        feedback_system = FeedbackSystem(db.session)
        
        learning_stats = learner.get_learning_stats()
        feedback_stats = feedback_system.get_feedback_stats()
        
        # Get scheduler status
        scheduler = get_learning_scheduler()
        scheduler_status = scheduler.get_scheduler_status() if scheduler else {}
        
        return render_template('admin/learning_dashboard.html', 
                             learning_stats={
                                 'learning_stats': learning_stats,
                                 'feedback_stats': feedback_stats,
                                 'scheduler_status': scheduler_status
                             })
        
    except Exception as e:
        flash(f'Error loading learning dashboard: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))

@app.route('/admin/trigger-learning', methods=['POST'])
@login_required
def admin_trigger_learning():
    """Trigger immediate model learning"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    try:
        data = request.get_json()
        learning_type = data.get('learning_type', 'all')
        
        learner = ModelLearner(db.session)
        
        if learning_type == 'form':
            success = learner.incremental_learning('form')
        elif learning_type == 'image':
            success = learner.incremental_learning('image')
        elif learning_type == 'all':
            success = learner.incremental_learning('form') and learner.incremental_learning('image')
        else:
            return jsonify({'error': 'Invalid learning type'}), 400
        
        if success:
            return jsonify({'success': True, 'message': f'{learning_type} learning completed successfully'})
        else:
            return jsonify({'error': f'{learning_type} learning failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/scheduler-control', methods=['POST'])
@login_required
def admin_scheduler_control():
    """Control the learning scheduler"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    try:
        data = request.get_json()
        action = data.get('action')
        
        scheduler = get_learning_scheduler()
        if not scheduler:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
        if action == 'start':
            scheduler.start_scheduler()
            return jsonify({'success': True, 'message': 'Scheduler started'})
        elif action == 'stop':
            scheduler.stop_scheduler()
            return jsonify({'success': True, 'message': 'Scheduler stopped'})
        elif action == 'enable':
            scheduler.enable_learning()
            return jsonify({'success': True, 'message': 'Learning enabled'})
        elif action == 'disable':
            scheduler.disable_learning()
            return jsonify({'success': True, 'message': 'Learning disabled'})
        else:
            return jsonify({'error': 'Invalid action'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/predictions-needing-feedback')
@login_required
def admin_predictions_needing_feedback():
    """Get predictions that need user feedback"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    try:
        feedback_system = FeedbackSystem(db.session)
        predictions = feedback_system.get_predictions_needing_feedback(limit=20)
        
        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                'id': pred.id,
                'type': pred.prediction_type,
                'result': pred.prediction_result,
                'confidence': pred.confidence,
                'created_at': pred.created_at.isoformat(),
                'image_url': url_for('uploaded_file', filename=pred.image_filename) if pred.image_filename else None
            })
        
        return jsonify({'predictions': prediction_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/submit-feedback/<int:prediction_id>', methods=['POST'])
@login_required
def admin_submit_feedback(prediction_id):
    """Submit feedback on a prediction"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin privileges required'}), 403
    
    try:
        feedback_type = request.form.get('feedback_type')
        feedback_comment = request.form.get('feedback_comment', '')
        
        if not feedback_type:
            return jsonify({'error': 'Feedback type is required'}), 400
        
        feedback_system = FeedbackSystem(db.session)
        success, message = feedback_system.submit_feedback(
            prediction_id, 
            feedback_type, 
            {'comment': feedback_comment},
            user_id=current_user.id,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        if success:
            return jsonify({'success': True, 'message': message})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Static file serving for uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded images"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API routes
@app.route('/api/stats')
@login_required
def api_stats():
    """API endpoint for statistics"""
    if current_user.is_admin:
        stats = get_prediction_stats()
        return jsonify(stats)
    else:
        return jsonify({'error': 'Admin access required'}), 403

# Learning System API Endpoints
@app.route('/api/learning/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """Submit feedback on a prediction"""
    try:
        data = request.get_json()
        prediction_id = data.get('prediction_id')
        feedback_type = data.get('feedback_type')  # 'correct', 'incorrect', 'uncertain'
        feedback_data = data.get('feedback_data', {})
        
        if not prediction_id or not feedback_type:
            return jsonify({'error': 'Missing required fields'}), 400
        
        if feedback_type not in ['correct', 'incorrect', 'uncertain']:
            return jsonify({'error': 'Invalid feedback type'}), 400
        
        # Submit feedback
        feedback_system = FeedbackSystem(db.session)
        success, message = feedback_system.submit_feedback(
            prediction_id, 
            feedback_type, 
            feedback_data,
            user_id=current_user.id if current_user.is_authenticated else None,
            ip_address=request.remote_addr,
            user_agent=request.headers.get('User-Agent')
        )
        
        if success:
            return jsonify({'message': message, 'success': True})
        else:
            return jsonify({'error': message}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/trigger', methods=['POST'])
@login_required
def trigger_learning():
    """Trigger immediate model learning"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        data = request.get_json()
        data_type = data.get('data_type', 'form')  # 'form' or 'image'
        
        if data_type not in ['form', 'image']:
            return jsonify({'error': 'Invalid data type'}), 400
        
        # Trigger learning
        learner = ModelLearner(db.session)
        success = learner.incremental_learning(data_type)
        
        if success:
            return jsonify({'message': f'{data_type} model learning completed successfully', 'success': True})
        else:
            return jsonify({'error': f'{data_type} model learning failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/stats')
@login_required
def learning_stats():
    """Get learning statistics"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        learner = ModelLearner(db.session)
        feedback_system = FeedbackSystem(db.session)
        
        learning_stats = learner.get_learning_stats()
        feedback_stats = feedback_system.get_feedback_stats()
        
        # Get scheduler status
        scheduler = get_learning_scheduler()
        scheduler_status = scheduler.get_scheduler_status() if scheduler else {}
        
        return jsonify({
            'learning_stats': learning_stats,
            'feedback_stats': feedback_stats,
            'scheduler_status': scheduler_status
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/predictions/needing-feedback')
@login_required
def predictions_needing_feedback():
    """Get predictions that need user feedback"""
    try:
        feedback_system = FeedbackSystem(db.session)
        predictions = feedback_system.get_predictions_needing_feedback(limit=20)
        
        prediction_data = []
        for pred in predictions:
            prediction_data.append({
                'id': pred.id,
                'prediction_type': pred.prediction_type,
                'prediction_result': pred.prediction_result,
                'confidence': pred.confidence,
                'created_at': pred.created_at.isoformat(),
                'image_url': url_for('uploaded_file', filename=pred.image_filename) if pred.image_filename else None
            })
        
        return jsonify({'predictions': prediction_data})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/learning/scheduler/control', methods=['POST'])
@login_required
def control_scheduler():
    """Control the learning scheduler"""
    if not current_user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    try:
        data = request.get_json()
        action = data.get('action')  # 'start', 'stop', 'enable', 'disable'
        
        scheduler = get_learning_scheduler()
        if not scheduler:
            return jsonify({'error': 'Scheduler not initialized'}), 500
        
        if action == 'start':
            scheduler.start_scheduler()
            return jsonify({'message': 'Scheduler started'})
        elif action == 'stop':
            scheduler.stop_scheduler()
            return jsonify({'message': 'Scheduler stopped'})
        elif action == 'enable':
            scheduler.enable_learning()
            return jsonify({'message': 'Learning enabled'})
        elif action == 'disable':
            scheduler.disable_learning()
            return jsonify({'message': 'Learning disabled'})
        else:
            return jsonify({'error': 'Invalid action'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    with app.app_context():
        init_db(app)
        
        # Initialize learning scheduler
        try:
            print("Initializing learning scheduler...")
            initialize_learning_scheduler(db.session)
            print("Learning scheduler initialized successfully!")
            print("Automatic learning system is ACTIVE - models will improve continuously!")
        except Exception as e:
            print(f"Warning: Learning scheduler initialization failed: {e}")
        
        # Check if models need retraining
        try:
            retrain_needed, reason = check_retraining_needed()
            if retrain_needed:
                print(f"Auto-retraining triggered: {reason}")
                result = auto_retrain_if_needed()
                print(f"Retraining result: {result}")
        except Exception as e:
            print(f"Auto-retraining check failed: {e}")
    
    print(" Starting  server...")
    print(f"Loaded {len(feature_names)} features")
    
    # Initialize CNN predictor
    try:
        print("Initializing CNN predictor...")
        initialize_cnn_predictor()
        print("CNN predictor initialized successfully!")
    except Exception as e:
        print(f"Warning: CNN predictor initialization failed: {e}")
        print("Image prediction will use fallback methods")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
