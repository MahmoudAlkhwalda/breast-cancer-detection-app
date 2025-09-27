#!/usr/bin/env python3
"""
Run the Breast Cancer Detection Application
"""

import os
import sys
from app import app, init_db, auto_retrain_if_needed, check_retraining_needed

def main():
    """Main function to run the application"""
    print("=" * 60)
    print("Breast Cancer Detection System")
    print("=" * 60)
    
    # Initialize database
    with app.app_context():
        print("Initializing database...")
        init_db(app)
        print("Database initialized successfully!")
        
        # Check for auto-retraining
        print("Checking for model retraining...")
        try:
            retrain_needed, reason = check_retraining_needed()
            if retrain_needed:
                print(f" Auto-retraining triggered: {reason}")
                result = auto_retrain_if_needed()
                print(f" Retraining result: {result}")
            else:
                print("ℹ️  No retraining needed at this time")
        except Exception as e:
            print(f"⚠️  Auto-retraining check failed: {e}")
    
    print("\nStarting server...")
    print("Server will be available at: http://localhost:5000")
    print("Admin login: admin / admin123")
    print("=" * 60)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()
