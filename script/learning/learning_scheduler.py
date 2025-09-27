"""
Learning Scheduler for Automatic Model Retraining
Schedules and manages automatic model learning from database data
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.cron import CronTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    print("Warning: APScheduler not available. Learning scheduler will be disabled.")
    APSCHEDULER_AVAILABLE = False
    # Create dummy classes for compatibility
    class BackgroundScheduler:
        def __init__(self, *args, **kwargs):
            pass
        def add_job(self, *args, **kwargs):
            pass
        def start(self):
            pass
        def shutdown(self):
            pass
        def get_jobs(self):
            return []
    class IntervalTrigger:
        def __init__(self, *args, **kwargs):
            pass
    class CronTrigger:
        def __init__(self, *args, **kwargs):
            pass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class LearningScheduler:
    def __init__(self, db_session=None):
        """
        Initialize the learning scheduler
        """
        self.db_session = db_session
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        self.learning_interval_hours = 24  # Learn every 24 hours
        self.min_samples_for_learning = 50
        self.learning_enabled = True
        
    def start_scheduler(self):
        """
        Start the learning scheduler
        """
        try:
            if not APSCHEDULER_AVAILABLE:
                print("APScheduler not available. Learning scheduler disabled.")
                return
                
            if not self.is_running:
                # Schedule daily learning
                self.scheduler.add_job(
                    func=self.scheduled_learning,
                    trigger=CronTrigger(hour=2, minute=0),  # Run at 2 AM daily
                    id='daily_learning',
                    name='Daily Model Learning',
                    replace_existing=True
                )
                
                # Schedule weekly full retraining
                self.scheduler.add_job(
                    func=self.scheduled_full_retraining,
                    trigger=CronTrigger(day_of_week=0, hour=3, minute=0),  # Run on Sunday at 3 AM
                    id='weekly_retraining',
                    name='Weekly Full Model Retraining',
                    replace_existing=True
                )
                
                # Schedule learning check every 6 hours
                self.scheduler.add_job(
                    func=self.check_learning_opportunities,
                    trigger=IntervalTrigger(hours=6),
                    id='learning_check',
                    name='Learning Opportunities Check',
                    replace_existing=True
                )
                
                self.scheduler.start()
                self.is_running = True
                print("Learning scheduler started successfully!")
                
        except Exception as e:
            print(f"Error starting learning scheduler: {e}")
    
    def stop_scheduler(self):
        """
        Stop the learning scheduler
        """
        try:
            if not APSCHEDULER_AVAILABLE:
                print("APScheduler not available. Learning scheduler already disabled.")
                return
                
            if self.is_running:
                self.scheduler.shutdown()
                self.is_running = False
                print("Learning scheduler stopped")
        except Exception as e:
            print(f"Error stopping learning scheduler: {e}")
    
    def scheduled_learning(self):
        """
        Scheduled learning task
        """
        try:
            print(f"Starting scheduled learning at {datetime.now()}")
            
            if not self.learning_enabled:
                print("Learning is disabled, skipping scheduled learning")
                return
            
            # Use automatic learning system
            from script.learning.automatic_learner import trigger_automatic_learning
            
            success = trigger_automatic_learning(self.db_session)
            
            if success:
                print("Scheduled automatic learning completed successfully")
            else:
                print("Scheduled automatic learning failed")
            
            print(f"Scheduled learning completed at {datetime.now()}")
            
        except Exception as e:
            print(f"Error in scheduled learning: {e}")
    
    def scheduled_full_retraining(self):
        """
        Scheduled full retraining task
        """
        try:
            print(f"Starting scheduled full retraining at {datetime.now()}")
            
            if not self.learning_enabled:
                print("Learning is disabled, skipping scheduled retraining")
                return
            
            from script.learning.model_learner import ModelLearner
            
            learner = ModelLearner(self.db_session)
            
            # Get learning stats
            stats = learner.get_learning_stats()
            print(f"Learning stats: {stats}")
            
            # Check if we have enough data for full retraining
            if stats.get('verified_form_predictions', 0) >= 100:
                print("Performing full form model retraining...")
                success = learner.incremental_learning('form')
                if success:
                    print("Full form model retraining completed")
                else:
                    print("Full form model retraining failed")
            
            if stats.get('verified_image_predictions', 0) >= 100:
                print("Performing full image model retraining...")
                success = learner.incremental_learning('image')
                if success:
                    print("Full image model retraining completed")
                else:
                    print("Full image model retraining failed")
            
            print(f"Scheduled full retraining completed at {datetime.now()}")
            
        except Exception as e:
            print(f"Error in scheduled full retraining: {e}")
    
    def check_learning_opportunities(self):
        """
        Check for learning opportunities
        """
        try:
            if not self.learning_enabled:
                return
            
            from script.learning.model_learner import ModelLearner
            
            learner = ModelLearner(self.db_session)
            
            # Check form data
            if learner.should_learn('form'):
                print("Learning opportunity detected for form data")
                # Could trigger immediate learning here if desired
            
            # Check image data
            if learner.should_learn('image'):
                print("Learning opportunity detected for image data")
                # Could trigger immediate learning here if desired
                
        except Exception as e:
            print(f"Error checking learning opportunities: {e}")
    
    def trigger_immediate_learning(self, data_type='form'):
        """
        Trigger immediate learning
        """
        try:
            print(f"Triggering immediate learning for {data_type} data...")
            
            from script.learning.model_learner import ModelLearner
            
            learner = ModelLearner(self.db_session)
            success = learner.incremental_learning(data_type)
            
            if success:
                print(f"Immediate learning completed for {data_type} data")
            else:
                print(f"Immediate learning failed for {data_type} data")
            
            return success
            
        except Exception as e:
            print(f"Error in immediate learning: {e}")
            return False
    
    def get_scheduler_status(self):
        """
        Get scheduler status
        """
        try:
            if not APSCHEDULER_AVAILABLE:
                return {
                    'is_running': False,
                    'learning_enabled': self.learning_enabled,
                    'jobs': [],
                    'apscheduler_available': False
                }
            
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run_time': job.next_run_time.isoformat() if job.next_run_time else None,
                    'trigger': str(job.trigger)
                })
            
            return {
                'is_running': self.is_running,
                'learning_enabled': self.learning_enabled,
                'jobs': jobs,
                'apscheduler_available': True
            }
            
        except Exception as e:
            print(f"Error getting scheduler status: {e}")
            return {'is_running': False, 'learning_enabled': False, 'jobs': [], 'apscheduler_available': False}
    
    def enable_learning(self):
        """
        Enable learning
        """
        self.learning_enabled = True
        print("Learning enabled")
    
    def disable_learning(self):
        """
        Disable learning
        """
        self.learning_enabled = False
        print("Learning disabled")

# Global scheduler instance
learning_scheduler = None

def initialize_learning_scheduler(db_session):
    """
    Initialize the global learning scheduler
    """
    global learning_scheduler
    learning_scheduler = LearningScheduler(db_session)
    learning_scheduler.start_scheduler()
    return learning_scheduler

def get_learning_scheduler():
    """
    Get the global learning scheduler
    """
    return learning_scheduler

if __name__ == "__main__":
    # Test the learning scheduler
    print("Learning Scheduler Test")
    print("=" * 30)
    
    scheduler = LearningScheduler()
    print("Learning scheduler initialized successfully!")
    
    # This would be used with a proper database session
    # scheduler = initialize_learning_scheduler(db_session)
