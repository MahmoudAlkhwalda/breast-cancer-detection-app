"""
Configuration settings for the Breast Cancer Analysis Application
"""

import os

class Config:
    """Base configuration class"""
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///breast_cancer_app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    UPLOAD_FOLDER = os.path.join('uploads', 'images')
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
    
    # Image processing settings
    IMAGE_QUALITY = 95
    IMAGE_COMPRESSION = True
    THUMBNAIL_SIZE = (300, 300)
    PREVIEW_SIZE = (600, 600)
    
    ML_MODEL_PATH = os.path.join('models', 'breast_cancer_model.pkl')
    CNN_MODEL_PATH = os.path.join('models', 'cnn_breast_cancer_model.h5')
    
    IMAGE_SIZE = (128, 128)
    IMAGE_CHANNELS = 3
    
    CNN_BATCH_SIZE = 32
    CNN_EPOCHS = 20
    CNN_LEARNING_RATE = 0.0001
    
    DATA_DIR = os.path.join('data')
    TRAIN_DATA_DIR = os.path.join('data', 'train')
    VALIDATION_DATA_DIR = os.path.join('data', 'validation')
    TEST_DATA_DIR = os.path.join('data', 'test')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
