# Breast Cancer Analysis Application

A comprehensive Flask web application for breast cancer analysis using both traditional machine learning and deep learning (CNN) approaches.

## 🏗️ Project Structure

```
breast_cancer_app/
├── app.py                          # Main Flask application
├── config.py                       # Configuration settings
├── requirements.txt                # Python dependencies
├── PROJECT_STRUCTURE.md           # Detailed project structure
│
├── data/                          # Dataset directory (BUSI dataset)
│   ├── train/                     # Training images (1,122 total)
│   │   ├── benign/               # Benign tumor images (627 files)
│   │   ├── malignant/            # Malignant tumor images (309 files)
│   │   └── normal/               # Normal tissue images (186 files)
│   ├── validation/               # Validation images (239 total)
│   │   ├── benign/               # Benign validation images (134 files)
│   │   ├── malignant/            # Malignant validation images (66 files)
│   │   └── normal/               # Normal validation images (39 files)
│   └── test/                     # Test images (244 total)
│       ├── benign/               # Benign test images (137 files)
│       ├── malignant/            # Malignant test images (68 files)
│       └── normal/               # Normal test images (39 files)
│
├── models/                        # Trained models
│   ├── best_cnn_model.h5         # Best performing CNN model
│   ├── cnn_breast_cancer_trained.h5  # Trained CNN model
│   └── breast_cancer_model.pkl   # RandomForest model for form predictions
│
├── scripts/                       # Training and utility scripts
│   ├── train_cnn_with_dataset.py # CNN training script
│   ├── train_improved_cnn.py     # Improved CNN training
│   └── organize_busi_dataset.py  # Dataset organization utility
│
├── src/                          # Source code
│   ├── cnn/                      # CNN-related modules
│   │   ├── advanced_cnn_predictor.py  # Advanced CNN with ensemble
│   │   ├── cnn_predictor.py      # Main CNN predictor
│   │   ├── cnn.py               # Basic CNN implementation
│   │   ├── pretrained_cnn.py    # Pre-trained model wrapper
│   │   └── train_cnn.py         # Legacy training script
│   ├── ml_models/               # Machine learning models
│   │   ├── model.py             # Model loading and prediction
│   │   └── train_model.py       # Model training utilities
│   └── models/                  # Additional model files
│       └── breast_cancer_model.pkl  # Duplicate model file
│
├── static/                       # Static web assets
│   ├── css/
│   │   └── style.css            # Application styles
│   ├── js/
│   │   └── script.js            # Client-side JavaScript
│   └── uploads/                 # User uploaded images
│
├── templates/                    # HTML templates
│   ├── base.html               # Base template
│   ├── index.html              # Home page
│   ├── form_predict.html       # Form prediction page
│   ├── image_predict.html      # Image prediction page
│   ├── results.html            # Results display
│   └── error.html              # Error page
│
├── tests/                       # Test files
│   ├── test_cnn_integration.py # CNN integration tests
│   ├── test_form_prediction.py # Form prediction tests
│   └── test_realistic_prediction.py # Realistic prediction tests
│
├── docs/                        # Documentation
│   ├── README.md               # Main documentation
│   ├── CNN_IMPROVEMENTS.md     # CNN improvement notes
│   ├── CNN_INTEGRATION_README.md # CNN integration guide
│   ├── FORM_PREDICTION_FIX.md  # Form prediction fixes
│   └── academic_breast_cancer_report.tex # Academic report
│
└── temp/                        # Temporary files
    ├── best_cancer_model.ipynb # Moved notebook
    └── breast_cancer_model_old.pkl # Old model file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Access the Application
Open your browser and go to: `http://localhost:5000`

## 🧠 Features

### Traditional Machine Learning
- **Random Forest Classifier** for numerical feature analysis
- **30 medical features** from breast cancer dataset
- **Form-based prediction** interface

### Deep Learning (CNN)
- **Convolutional Neural Network** for image analysis
- **3-class medical image classification** (Benign, Malignant, Normal)
- **Ensemble of pre-trained models** (VGG16, ResNet50, MobileNetV2)
- **Medical feature analysis** (intensity, variance, edge detection, LBP, entropy)
- **Bias correction** with temperature scaling
- **Image upload and prediction** interface
- **Fallback prediction** when no trained model is available

### Web Interface
- **Responsive design** with modern UI
- **Dual prediction modes**: Form input and image upload
- **Real-time results** with confidence scores
- **Error handling** and user feedback

## 📊 Usage

### Form-Based Prediction
1. Navigate to "Form Prediction"
2. Enter medical feature values
3. Get instant prediction with confidence score

### Image-Based Prediction
1. Navigate to "Image Prediction"
2. Upload a medical image (PNG, JPG, JPEG, BMP, TIFF)
3. Get CNN-based analysis with confidence score

## 🔧 Development

### Training Models

#### Traditional ML Model
```bash
python src/ml_models/train_model.py
```

#### CNN Model (Improved)
```bash
cd scripts
python train_improved_cnn.py
```

#### CNN Model (Basic)
```bash
cd scripts
python train_cnn_with_dataset.py
```

#### Organize Dataset
```bash
cd scripts
python organize_busi_dataset.py
```

### Running Tests
```bash
python tests/test_cnn_integration.py
```

## 📁 File Organization

- **`app.py`**: Main Flask application with all routes
- **`src/ml_models/`**: Traditional machine learning implementation
- **`src/cnn/`**: Deep learning CNN implementation
- **`static/`**: Web assets (CSS, JS, uploaded images)
- **`templates/`**: HTML templates for web interface
- **`models/`**: Trained model files
- **`tests/`**: Test scripts and validation
- **`docs/`**: Documentation and reports

## 🛠️ Configuration

### Environment Setup
- Python 3.8+
- Flask 2.3.3
- TensorFlow 2.13.0
- scikit-learn 1.3.0

### Model Paths
- Traditional ML model: `models/breast_cancer_model.pkl`
- CNN model: `models/cnn_breast_cancer_model.h5` (after training)

## 📈 Performance

- **Traditional ML**: High accuracy on numerical features
- **CNN**: Deep learning analysis of medical images
- **Fallback**: Heuristic-based prediction when models unavailable
- **Real-time**: Fast prediction response times

## 🔒 Security

- File upload validation
- Secure filename handling
- Input sanitization
- Error handling and logging

## 📝 Documentation

- **`docs/CNN_INTEGRATION_README.md`**: Detailed CNN integration guide
- **`docs/academic_breast_cancer_report.tex`**: Academic research report

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure proper medical validation for clinical use.

## 🆘 Support

For issues or questions:
1. Check the documentation in `docs/`
2. Run the test suite
3. Check console output for error messages
4. Verify all dependencies are installed

---

**Note**: This application is designed for educational and research purposes. For clinical use, ensure proper medical validation and regulatory compliance.
