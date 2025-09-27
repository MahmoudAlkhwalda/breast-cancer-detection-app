#!/usr/bin/env python3
"""Database management tools"""

from app import app, db, User, Prediction, UploadedImage, ModelTrainingData, ModelVersion, SystemLog
import sqlite3
import os

def export_database_to_csv():
    """Export database to CSV files"""
    with app.app_context():
        import pandas as pd
        
        # Export users
        users = User.query.all()
        users_data = []
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'created_at': user.created_at,
                'last_login': user.last_login
            })
        
        if users_data:
            df_users = pd.DataFrame(users_data)
            df_users.to_csv('database_export_users.csv', index=False)
            print("✅ Users exported to database_export_users.csv")
        
        # Export predictions
        predictions = Prediction.query.all()
        pred_data = []
        for pred in predictions:
            pred_data.append({
                'id': pred.id,
                'user_id': pred.user_id,
                'prediction_type': pred.prediction_type,
                'prediction_result': pred.prediction_result,
                'confidence': pred.confidence,
                'is_verified': pred.is_verified,
                'verified_result': pred.verified_result,
                'created_at': pred.created_at
            })
        
        if pred_data:
            df_pred = pd.DataFrame(pred_data)
            df_pred.to_csv('database_export_predictions.csv', index=False)
            print("✅ Predictions exported to database_export_predictions.csv")

def view_database_schema():
    """View database schema"""
    db_path = 'instance/breast_cancer_app.db'
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("=" * 60)
        print("🗄️ DATABASE SCHEMA")
        print("=" * 60)
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"\n📋 TABLE: {table_name}")
            print("-" * 40)
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            
            for col in columns:
                col_id, name, data_type, not_null, default, pk = col
                print(f"  {name}: {data_type} {'(PRIMARY KEY)' if pk else ''} {'NOT NULL' if not_null else ''}")
        
        conn.close()
    else:
        print("❌ Database file not found!")

def backup_database():
    """Create database backup"""
    import shutil
    from datetime import datetime
    
    db_path = 'instance/breast_cancer_app.db'
    if os.path.exists(db_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f'backup_breast_cancer_app_{timestamp}.db'
        shutil.copy2(db_path, backup_path)
        print(f"✅ Database backed up to: {backup_path}")
    else:
        print("❌ Database file not found!")

def clear_old_data():
    """Clear old data (be careful!)"""
    with app.app_context():
        print("⚠️  WARNING: This will clear old data!")
        print("Available options:")
        print("1. Clear old predictions (older than 30 days)")
        print("2. Clear old system logs (older than 7 days)")
        print("3. Clear all non-admin users")
        print("4. Clear all data (DANGEROUS!)")
        
        choice = input("Enter choice (1-4) or 'cancel': ")
        
        if choice == '1':
            from datetime import datetime, timedelta
            old_date = datetime.utcnow() - timedelta(days=30)
            old_predictions = Prediction.query.filter(Prediction.created_at < old_date).all()
            for pred in old_predictions:
                db.session.delete(pred)
            db.session.commit()
            print(f"✅ Cleared {len(old_predictions)} old predictions")
            
        elif choice == '2':
            from datetime import datetime, timedelta
            old_date = datetime.utcnow() - timedelta(days=7)
            old_logs = SystemLog.query.filter(SystemLog.created_at < old_date).all()
            for log in old_logs:
                db.session.delete(log)
            db.session.commit()
            print(f"✅ Cleared {len(old_logs)} old logs")
            
        elif choice == '3':
            non_admin_users = User.query.filter_by(is_admin=False).all()
            for user in non_admin_users:
                db.session.delete(user)
            db.session.commit()
            print(f"✅ Cleared {len(non_admin_users)} non-admin users")
            
        elif choice == '4':
            print("❌ Option 4 is too dangerous and disabled!")
            
        elif choice.lower() == 'cancel':
            print("✅ Operation cancelled")
        else:
            print("❌ Invalid choice")

if __name__ == '__main__':
    print("Database Management Tools")
    print("1. View database schema")
    print("2. Export to CSV")
    print("3. Backup database")
    print("4. Clear old data")
    
    choice = input("Enter choice (1-4): ")
    
    if choice == '1':
        view_database_schema()
    elif choice == '2':
        export_database_to_csv()
    elif choice == '3':
        backup_database()
    elif choice == '4':
        clear_old_data()
    else:
        print("❌ Invalid choice")
