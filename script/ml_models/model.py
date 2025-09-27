import numpy as np
import pandas as pd
import joblib
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import sys
import os
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y
def train_and_save_model():
    X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    model_data = {
        "model": model,
        "scaler": scaler,
        "feature_names": X.columns.tolist()
    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join('..', '..', 'models', 'breast_cancer_model.pkl')
    joblib.dump(model_data, model_path)
    print(f"Model saved to {model_path}")
    
    return model_data

def load_model():
    try:
        # Get the absolute path to the model file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'models', 'breast_cancer_model.pkl')
        data = joblib.load(model_path)
        return data
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model_data = load_model()
if model_data:
    model = model_data["model"]
    scaler = model_data["scaler"]
    feature_names = model_data["feature_names"]
    print(" Model loaded successfully!")
    print(f" Number of features: {len(feature_names)}")
else:
    print(" Failed to load model")
    model = None
    scaler = None
    feature_names = []

def predict_from_form(data_array):
    try:
        if model is None:
            print(" Model not loaded for prediction")
            return "Error", 0.5
        
        # Ensure data is in correct format
        if len(data_array.shape) == 1:
            data_array = data_array.reshape(1, -1)
        
        # Validate input data
        if data_array.shape[1] != len(feature_names):
            print(f" Error: Expected {len(feature_names)} features, got {data_array.shape[1]}")
            raise ValueError(f"Expected {len(feature_names)} features, got {data_array.shape[1]}")
        
        # Handle NaN or infinite values
        data_array = np.nan_to_num(data_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        data_df = pd.DataFrame(data_array, columns=feature_names)
        print(f" Input data shape: {data_df.shape}")
        
        data_scaled = scaler.transform(data_df)
        
        prediction = model.predict(data_scaled)
        proba = model.predict_proba(data_scaled)
        
        print(f"Prediction: {prediction[0]}")
        print(f" Confidence: {proba[0]}")
        
        # In sklearn's breast_cancer dataset: 0 = malignant, 1 = benign
        if prediction[0] == 0:
            confidence = float(proba[0][0])
            # Apply confidence adjustment for better balance
            confidence = min(0.95, confidence * 1.1)  # Slight boost for malignant
            return "Malignant", confidence
        else:
            confidence = float(proba[0][1])
            # Apply confidence adjustment for better balance
            confidence = min(0.95, confidence * 1.05)  # Slight boost for benign
            return "Benign", confidence
    
    except Exception as e:
        print(f" Error in prediction: {e}")
        raise e

# Note: Image prediction is now handled by the unified CNN predictor
# This function is kept for backward compatibility but redirects to the unified predictor
def predict_from_image(img_path):
    try:
        print(f" Image prediction for: {img_path}")
        
        # Import here to avoid circular imports
        from script.cnn.unified_cnn_predictor import get_cnn_prediction
        
        # Use unified CNN for image prediction
        prediction, confidence, message = get_cnn_prediction(img_path)
        
        print(f"CNN Prediction: {prediction}, Confidence: {confidence:.2%}")
        print(f"Message: {message}")
        
        return prediction, float(confidence)
    
    except Exception as e:
        print(f" Error in image prediction: {e}")
        return "Error", 0.5

if __name__ == "__main__":
    if model is None:
        print("Training model...")
        train_and_save_model()
        print(" Model training completed!")
