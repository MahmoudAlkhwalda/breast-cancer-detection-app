# 🏥 Breast Cancer Detection Application

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)](https://flask.palletsprojects.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen.svg)]()

A comprehensive Flask web application for breast cancer analysis using both traditional machine learning and deep learning (CNN) approaches.

## ✨ Features

### 🔬 Dual Prediction Methods
- **Form-based**: Enter 30 medical features for instant diagnosis
- **Image-based**: Upload medical images (PNG, JPG, JPEG, BMP, TIFF) for CNN analysis

### 🧠 AI/ML Capabilities
- **Traditional ML**: Random Forest Classifier with 30 medical features
- **Deep Learning**: CNN with ensemble of pre-trained models (VGG16, ResNet50, MobileNetV2)
- **3-class Classification**: Benign, Malignant, Normal
- **Real-time Predictions**: Instant results with confidence scores

### 🌐 Web Application
- **Responsive Design**: Modern, mobile-friendly interface
- **User Authentication**: Login/registration with admin privileges
- **Admin Dashboard**: Prediction management and model training
- **Learning System**: Continuous model improvement from user feedback

### 📊 Dataset & Performance
- **BUSI Dataset**: 1,122 training images (627 benign, 309 malignant, 186 normal)
- **High Accuracy**: Both ML and CNN models achieve high accuracy
- **Real-time Processing**: Fast prediction response times
- **Scalable**: Multiple concurrent users supported

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR-USERNAME/breast-cancer-detection-app.git
cd breast-cancer-detection-app
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
python app.py
```

5. **Access the application**
Open your browser and go to: `http://localhost:5000`

## 🛠️ Tech Stack

- **Backend**: Flask 2.3.3, SQLite
- **ML/DL**: TensorFlow 2.13.0, scikit-learn 1.3.0
- **Image Processing**: OpenCV 4.8.0, PIL/Pillow
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLAlchemy, Flask-Login

## 📁 Project Structure

```
breast_cancer_app/
├── app.py                    # Main Flask application
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── database_tools.py         # Database utilities
├── run_app.py               # Application runner
│
├── script/                  # Source code modules
│   ├── cnn/                 # CNN implementation
│   │   ├── cnn.py
│   │   ├── unified_cnn_predictor.py
│   │   └── unified_cnn_trainer.py
│   ├── ml_models/           # Traditional ML
│   │   ├── model.py
│   │   └── train_model.py
│   └── learning/            # Learning system
│       ├── automatic_learner.py
│       ├── feedback_system.py
│       ├── learning_scheduler.py
│       └── model_learner.py
│
├── data/                    # BUSI dataset
│   ├── train/              # Training images
│   ├── validation/         # Validation images
│   └── test/               # Test images
│
├── models/                  # Trained models
├── templates/               # HTML templates
├── static/                  # CSS/JS assets
├── tests/                   # Test files
└── docs/                    # Documentation
```

## 🎯 Use Cases

- 🎓 **Educational**: Learn ML in medical diagnosis
- 🔬 **Research**: Academic and research purposes
- 🏥 **Prototype**: Base for clinical diagnostic tools
- 📈 **Analysis**: Breast cancer pattern analysis

## 🔧 Development

### Training Models
```bash
# Traditional ML Model
python script/ml_models/train_model.py

# CNN Model (use unified training system)
# Automatic learning is built-in
```

### Running Tests
```bash
python tests/test_cnn_integration.py
python tests/test_form_prediction.py
```

## 📊 Performance Metrics

- **Traditional ML**: High accuracy on numerical features
- **CNN**: Deep learning analysis of medical images
- **Fallback**: Heuristic-based prediction when models unavailable
- **Real-time**: Fast prediction response times
- **Scalable**: Multiple concurrent users supported

## 🔒 Security Features

- File upload validation and sanitization
- Secure filename handling
- Input validation and sanitization
- Comprehensive error handling and logging
- User authentication and authorization
- Admin privilege management

## ⚠️ Important Notes

**Educational Purpose Only** - This application is designed for educational and research purposes. It is NOT intended for clinical use. Always consult medical professionals for actual diagnosis.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions or support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review the test files for usage examples

## 🙏 Acknowledgments

- BUSI Dataset for medical images
- TensorFlow and scikit-learn communities
- Flask framework contributors

---

**Status**: Active Development | **Version**: 1.0 | **Python**: 3.8+

Made with ❤️ for medical AI research
