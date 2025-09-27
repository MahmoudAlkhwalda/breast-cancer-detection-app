"""
Integration test for form prediction functionality
"""

import unittest
import os
import sys
import json
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, db, feature_names

class TestFormIntegration(unittest.TestCase):
    """Test form prediction integration with Flask app"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.app = app
        self.app.config['TESTING'] = True
        self.app.config['WTF_CSRF_ENABLED'] = False
        self.client = self.app.test_client()
        
        # Create test database
        with self.app.app_context():
            db.create_all()
    
    def tearDown(self):
        """Clean up after tests"""
        with self.app.app_context():
            db.drop_all()
    
    def test_form_page_loads(self):
        """Test that the form page loads correctly"""
        response = self.client.get('/form-predict')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Form-Based Prediction', response.data)
        self.assertIn(b'Predict', response.data)
    
    def test_form_submission_with_valid_data(self):
        """Test form submission with valid data"""
        # Create test data
        form_data = {}
        for i, feature in enumerate(feature_names):
            form_data[feature] = str(10.0 + i * 0.1)  # Different values for each feature
        
        response = self.client.post('/predict-form', data=form_data)
        
        # Should redirect to results page
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'prediction', response.data.lower())
    
    def test_form_submission_with_missing_data(self):
        """Test form submission with missing data"""
        # Create incomplete form data
        form_data = {}
        for i, feature in enumerate(feature_names[:15]):  # Only half the features
            form_data[feature] = str(10.0 + i * 0.1)
        
        response = self.client.post('/predict-form', data=form_data)
        
        # Should still work but with warnings
        self.assertEqual(response.status_code, 200)
    
    def test_form_submission_with_invalid_data(self):
        """Test form submission with invalid data"""
        # Create form data with invalid values
        form_data = {}
        for i, feature in enumerate(feature_names):
            if i % 3 == 0:
                form_data[feature] = "invalid_value"
            else:
                form_data[feature] = str(10.0 + i * 0.1)
        
        response = self.client.post('/predict-form', data=form_data)
        
        # Should still work but with warnings
        self.assertEqual(response.status_code, 200)
    
    def test_form_submission_with_empty_data(self):
        """Test form submission with empty data"""
        # Create empty form data
        form_data = {}
        
        response = self.client.post('/predict-form', data=form_data)
        
        # Should still work but with warnings
        self.assertEqual(response.status_code, 200)
    
    def test_form_validation(self):
        """Test form validation"""
        # Test with valid data
        form_data = {}
        for i, feature in enumerate(feature_names):
            form_data[feature] = str(10.0 + i * 0.1)
        
        response = self.client.post('/predict-form', data=form_data)
        self.assertEqual(response.status_code, 200)
        
        # Test with negative values (should still work)
        form_data = {}
        for i, feature in enumerate(feature_names):
            form_data[feature] = str(-10.0 + i * 0.1)
        
        response = self.client.post('/predict-form', data=form_data)
        self.assertEqual(response.status_code, 200)
    
    def test_form_features_count(self):
        """Test that form has correct number of features"""
        response = self.client.get('/form-predict')
        self.assertEqual(response.status_code, 200)
        
        # Count input fields in the response
        input_count = response.data.count(b'type="number"')
        self.assertEqual(input_count, len(feature_names))

if __name__ == '__main__':
    unittest.main()
