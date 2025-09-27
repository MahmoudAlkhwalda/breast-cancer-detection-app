"""
Feedback System for Model Learning
Allows users to provide feedback on predictions for model improvement
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class FeedbackSystem:
    def __init__(self, db_session=None):
        """
        Initialize the feedback system
        """
        self.db_session = db_session
        
    def submit_feedback(self, prediction_id, feedback_type, feedback_data, user_id=None, ip_address=None, user_agent=None):
        """
        Submit user feedback on a prediction
        
        Args:
            prediction_id: ID of the prediction
            feedback_type: Type of feedback ('correct', 'incorrect', 'uncertain')
            feedback_data: Additional feedback data
            user_id: ID of the user (optional)
            ip_address: IP address (optional)
            user_agent: User agent (optional)
        """
        try:
            from app import Prediction, SystemLog
            
            # Get the prediction
            prediction = self.db_session.query(Prediction).get(prediction_id)
            if not prediction:
                return False, "Prediction not found"
            
            # Update prediction with feedback
            if feedback_type == 'correct':
                prediction.is_verified = True
                prediction.verified_result = prediction.prediction_result
                prediction.verified_by = user_id or 'user'
                prediction.verified_at = datetime.utcnow()
                
                # Log the feedback
                log_message = f"User confirmed prediction as correct: {prediction.prediction_result}"
                
            elif feedback_type == 'incorrect':
                # User says prediction is wrong, but we need the correct result
                if 'correct_result' in feedback_data:
                    prediction.is_verified = True
                    prediction.verified_result = feedback_data['correct_result']
                    prediction.verified_by = user_id or 'user'
                    prediction.verified_at = datetime.utcnow()
                    
                    log_message = f"User corrected prediction: {prediction.prediction_result} -> {feedback_data['correct_result']}"
                else:
                    return False, "Correct result required for incorrect feedback"
                    
            elif feedback_type == 'uncertain':
                # User is uncertain about the prediction
                log_message = f"User marked prediction as uncertain: {prediction.prediction_result}"
                
            else:
                return False, "Invalid feedback type"
            
            # Add user notes if provided
            if 'notes' in feedback_data:
                if not prediction.form_data:
                    prediction.form_data = {}
                prediction.form_data['user_notes'] = feedback_data['notes']
                prediction.form_data['feedback_type'] = feedback_type
                prediction.form_data['feedback_date'] = datetime.utcnow().isoformat()
            
            # Save changes
            self.db_session.commit()
            
            # Log the feedback
            feedback_log = SystemLog(
                log_type='user_feedback',
                message=log_message,
                user_id=user_id,
                ip_address=ip_address,
                user_agent=user_agent,
                extra_data={
                    'prediction_id': prediction_id,
                    'feedback_type': feedback_type,
                    'feedback_data': feedback_data
                }
            )
            
            self.db_session.add(feedback_log)
            self.db_session.commit()
            
            return True, "Feedback submitted successfully"
            
        except Exception as e:
            self.db_session.rollback()
            return False, f"Error submitting feedback: {str(e)}"
    
    def get_feedback_stats(self):
        """
        Get feedback statistics
        """
        try:
            from app import Prediction, SystemLog
            
            stats = {
                'total_predictions': self.db_session.query(Prediction).count(),
                'verified_predictions': self.db_session.query(Prediction).filter_by(is_verified=True).count(),
                'unverified_predictions': self.db_session.query(Prediction).filter_by(is_verified=False).count(),
                'user_feedback_count': self.db_session.query(SystemLog).filter_by(log_type='user_feedback').count(),
                'correct_feedback': self.db_session.query(SystemLog).filter(
                    SystemLog.log_type == 'user_feedback',
                    SystemLog.message.like('%confirmed prediction as correct%')
                ).count(),
                'incorrect_feedback': self.db_session.query(SystemLog).filter(
                    SystemLog.log_type == 'user_feedback',
                    SystemLog.message.like('%corrected prediction%')
                ).count(),
                'uncertain_feedback': self.db_session.query(SystemLog).filter(
                    SystemLog.log_type == 'user_feedback',
                    SystemLog.message.like('%marked prediction as uncertain%')
                ).count()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            return {}
    
    def get_predictions_needing_feedback(self, limit=10):
        """
        Get predictions that need user feedback
        """
        try:
            from app import Prediction
            
            # Get recent unverified predictions
            predictions = self.db_session.query(Prediction).filter(
                Prediction.is_verified == False
            ).order_by(Prediction.created_at.desc()).limit(limit).all()
            
            return predictions
            
        except Exception as e:
            print(f"Error getting predictions needing feedback: {e}")
            return []
    
    def get_learning_candidates(self, data_type='form', min_samples=50):
        """
        Get predictions that are good candidates for learning
        """
        try:
            from app import Prediction
            
            # Get verified predictions for learning
            predictions = self.db_session.query(Prediction).filter(
                Prediction.prediction_type == data_type,
                Prediction.is_verified == True
            ).order_by(Prediction.verified_at.desc()).all()
            
            if len(predictions) >= min_samples:
                return predictions
            else:
                print(f"Not enough verified predictions for learning: {len(predictions)} < {min_samples}")
                return []
                
        except Exception as e:
            print(f"Error getting learning candidates: {e}")
            return []

def submit_prediction_feedback(db_session, prediction_id, feedback_type, feedback_data):
    """
    Convenience function to submit feedback
    """
    feedback_system = FeedbackSystem(db_session)
    return feedback_system.submit_feedback(prediction_id, feedback_type, feedback_data)

if __name__ == "__main__":
    print("Feedback System initialized successfully!")
