"""
Automatic Learning System - Works Behind the Scenes
Learns from all predictions automatically without user interaction
"""

import os
import sys
import numpy as np
import random
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class AutomaticLearner:
    def __init__(self, db_session=None):
        """
        Initialize the automatic learning system
        """
        self.db_session = db_session
        self.learning_threshold = 20  # Lower threshold for automatic learning
        self.retrain_threshold = 50   # Threshold for full retraining
        self.learning_enabled = True
        
    def should_learn_automatically(self, prediction):
        """
        Determine if we should learn from this prediction automatically
        """
        try:
            # Learn from low confidence predictions
            if prediction.confidence < 0.7:
                print(f"    📉 Low confidence detected: {prediction.confidence:.3f}")
                return True, "low_confidence"
            
            # Learn from predictions with specific patterns
            if prediction.prediction_type == 'form' and prediction.confidence < 0.8:
                print(f"    📝 Form low confidence: {prediction.confidence:.3f}")
                return True, "form_low_confidence"
            
            if prediction.prediction_type == 'image' and prediction.confidence < 0.75:
                print(f"    🖼️ Image low confidence: {prediction.confidence:.3f}")
                return True, "image_low_confidence"
            
            # Learn from certain prediction results
            if prediction.prediction_result in ['Malignant', 'Normal']:
                print(f"    🎯 Important result: {prediction.prediction_result}")
                return True, "specific_result"
            
            # Random learning (10% chance)
            if random.random() < 0.1:
                print(f"    🎲 Random learning triggered")
                return True, "random_learning"
            
            print(f"    ⏭️ No learning needed (conf: {prediction.confidence:.3f})")
            return False, "no_learning_needed"
            
        except Exception as e:
            print(f"❌ Error in should_learn_automatically: {e}")
            return False, "error"
    
    def create_synthetic_feedback(self, prediction):
        """
        Create synthetic feedback based on prediction characteristics
        """
        try:
            # Simulate user feedback based on confidence and result
            if prediction.confidence > 0.9:
                # High confidence predictions are likely correct
                feedback_type = 'correct'
                feedback_data = {}
                print(f"    ✅ High confidence - marking as correct")
            elif prediction.confidence < 0.6:
                # Low confidence predictions might be wrong
                if random.random() < 0.3:  # 30% chance of being wrong
                    feedback_type = 'incorrect'
                    # Generate a different result
                    possible_results = ['Benign', 'Malignant', 'Normal']
                    possible_results.remove(prediction.prediction_result)
                    correct_result = random.choice(possible_results)
                    feedback_data = {
                        'correct_result': correct_result,
                        'notes': 'Synthetic feedback based on low confidence'
                    }
                    print(f"    ❌ Low confidence - marking as incorrect: {prediction.prediction_result} → {correct_result}")
                else:
                    feedback_type = 'correct'
                    feedback_data = {}
                    print(f"    ✅ Low confidence - marking as correct")
            else:
                # Medium confidence - mostly correct
                if random.random() < 0.1:  # 10% chance of being wrong
                    feedback_type = 'incorrect'
                    possible_results = ['Benign', 'Malignant', 'Normal']
                    possible_results.remove(prediction.prediction_result)
                    correct_result = random.choice(possible_results)
                    feedback_data = {
                        'correct_result': correct_result,
                        'notes': 'Synthetic feedback based on medium confidence'
                    }
                    print(f"    ❌ Medium confidence - marking as incorrect: {prediction.prediction_result} → {correct_result}")
                else:
                    feedback_type = 'correct'
                    feedback_data = {}
                    print(f"    ✅ Medium confidence - marking as correct")
            
            return feedback_type, feedback_data
            
        except Exception as e:
            print(f"❌ Error creating synthetic feedback: {e}")
            return 'correct', {}
    
    def process_prediction_automatically(self, prediction):
        """
        Process a prediction for automatic learning
        """
        try:
            if not self.learning_enabled:
                print(f"    ⏸️ Learning disabled for prediction {prediction.id}")
                return False
            
            print(f"    🔍 Analyzing prediction {prediction.id} (conf: {prediction.confidence:.3f}, type: {prediction.prediction_type})")
            
            # Check if we should learn from this prediction
            should_learn, reason = self.should_learn_automatically(prediction)
            
            if not should_learn:
                return False
            
            print(f"    🚀 Learning triggered: {reason}")
            
            # Create synthetic feedback
            feedback_type, feedback_data = self.create_synthetic_feedback(prediction)
            
            # Apply the feedback
            from script.learning.feedback_system import FeedbackSystem
            feedback_system = FeedbackSystem(self.db_session)
            
            success, message = feedback_system.submit_feedback(
                prediction.id,
                feedback_type,
                feedback_data,
                user_id='automatic_learner',
                ip_address='127.0.0.1',
                user_agent='AutomaticLearner/1.0'
            )
            
            if success:
                print(f"    ✅ Learning applied: {feedback_type}")
                return True
            else:
                print(f"    ❌ Learning failed: {message}")
                return False
                
        except Exception as e:
            print(f"❌ Error in process_prediction_automatically: {e}")
            return False
    
    def learn_from_recent_predictions(self, hours=24):
        """
        Learn from recent predictions automatically
        """
        try:
            if not self.learning_enabled:
                return False
            
            from app import Prediction
            
            # Get recent predictions
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_predictions = self.db_session.query(Prediction).filter(
                Prediction.created_at >= cutoff_time,
                Prediction.is_verified == False  # Only unverified predictions
            ).all()
            
            print(f"Found {len(recent_predictions)} recent predictions for automatic learning")
            
            learned_count = 0
            for prediction in recent_predictions:
                if self.process_prediction_automatically(prediction):
                    learned_count += 1
            
            print(f"Automatic learning completed: {learned_count}/{len(recent_predictions)} predictions processed")
            return learned_count > 0
            
        except Exception as e:
            print(f"Error in learn_from_recent_predictions: {e}")
            return False
    
    def trigger_automatic_learning(self):
        """
        Trigger automatic learning from all available data
        """
        try:
            if not self.learning_enabled:
                print("Automatic learning is disabled")
                return False
            
            print("Starting automatic learning process...")
            
            # Learn from recent predictions
            recent_learning = self.learn_from_recent_predictions(hours=24)
            
            # Check if we have enough verified data for model retraining
            from script.learning.model_learner import ModelLearner
            learner = ModelLearner(self.db_session)
            
            # Try form learning
            if learner.should_learn('form'):
                print("Triggering automatic form model learning...")
                success = learner.incremental_learning('form')
                if success:
                    print("Automatic form model learning completed")
                else:
                    print("Automatic form model learning failed")
            
            # Try image learning
            if learner.should_learn('image'):
                print("Triggering automatic image model learning...")
                success = learner.incremental_learning('image')
                if success:
                    print("Automatic image model learning completed")
                else:
                    print("Automatic image model learning failed")
            
            print("Automatic learning process completed")
            return True
            
        except Exception as e:
            print(f"Error in trigger_automatic_learning: {e}")
            return False
    
    def get_automatic_learning_stats(self):
        """
        Get statistics about automatic learning
        """
        try:
            from app import SystemLog
            
            stats = {
                'automatic_learning_enabled': self.learning_enabled,
                'learning_threshold': self.learning_threshold,
                'retrain_threshold': self.retrain_threshold,
                'automatic_feedback_count': self.db_session.query(SystemLog).filter(
                    SystemLog.log_type == 'user_feedback',
                    SystemLog.user_id == 'automatic_learner'
                ).count(),
                'recent_automatic_feedback': self.db_session.query(SystemLog).filter(
                    SystemLog.log_type == 'user_feedback',
                    SystemLog.user_id == 'automatic_learner',
                    SystemLog.timestamp >= datetime.utcnow() - timedelta(hours=24)
                ).count()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting automatic learning stats: {e}")
            return {}

def process_prediction_for_learning(db_session, prediction):
    """
    Convenience function to process a prediction for automatic learning
    """
    learner = AutomaticLearner(db_session)
    return learner.process_prediction_automatically(prediction)

def trigger_automatic_learning(db_session):
    """
    Convenience function to trigger automatic learning
    """
    learner = AutomaticLearner(db_session)
    return learner.trigger_automatic_learning()

if __name__ == "__main__":
    print("Automatic Learning System initialized successfully!")
