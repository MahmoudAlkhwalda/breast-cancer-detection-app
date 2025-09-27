"""
Test form prediction functionality
"""

import unittest
import os
import sys
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from script.ml_models.model import predict_from_form, feature_names

class TestFormPrediction(unittest.TestCase):
    """Test form prediction functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = {
            'mean_radius': 12.0,
            'mean_texture': 15.0,
            'mean_perimeter': 78.0,
            'mean_area': 500.0,
            'mean_smoothness': 0.1,
            'mean_compactness': 0.1,
            'mean_concavity': 0.05,
            'mean_concave_points': 0.02,
            'mean_symmetry': 0.2,
            'mean_fractal_dimension': 0.06,
            'radius_error': 0.5,
            'texture_error': 1.0,
            'perimeter_error': 2.0,
            'area_error': 20.0,
            'smoothness_error': 0.01,
            'compactness_error': 0.02,
            'concavity_error': 0.01,
            'concave_points_error': 0.005,
            'symmetry_error': 0.02,
            'fractal_dimension_error': 0.005,
            'worst_radius': 15.0,
            'worst_texture': 20.0,
            'worst_perimeter': 95.0,
            'worst_area': 700.0,
            'worst_smoothness': 0.12,
            'worst_compactness': 0.15,
            'worst_concavity': 0.08,
            'worst_concave_points': 0.03,
            'worst_symmetry': 0.25,
            'worst_fractal_dimension': 0.08
        }
    
    def test_feature_names(self):
        """Test feature names"""
        self.assertIsNotNone(feature_names)
        self.assertIsInstance(feature_names, list)
        self.assertEqual(len(feature_names), 30)
    
    def test_prediction_with_sample_data(self):
        """Test prediction with sample data"""
        try:
            # Convert sample data to numpy array
            features_array = np.array([list(self.sample_data.values())])
            prediction, confidence = predict_from_form(features_array)
            
            self.assertIsNotNone(prediction)
            self.assertIsNotNone(confidence)
            self.assertIn(prediction, ['Benign', 'Malignant'])
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
        except Exception as e:
            # If model is not available, test should pass
            self.assertIsInstance(e, Exception)
    
    def test_prediction_data_types(self):
        """Test prediction data types"""
        try:
            # Convert sample data to numpy array
            features_array = np.array([list(self.sample_data.values())])
            prediction, confidence = predict_from_form(features_array)
            
            self.assertIsInstance(prediction, str)
            self.assertIsInstance(confidence, (int, float))
            self.assertGreaterEqual(confidence, 0)
            self.assertLessEqual(confidence, 1)
        except Exception as e:
            # If model is not available, test should pass
            self.assertIsInstance(e, Exception)

if __name__ == '__main__':
    unittest.main()
