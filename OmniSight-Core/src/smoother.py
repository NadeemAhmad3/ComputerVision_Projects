# src/smoother.py
from collections import deque

class OmniSmoother:
    def __init__(self, buffer_size=10):
        """
        Initializes the temporal smoothing engine.
        Uses consensus-based approach to reduce prediction jitter.
        
        Args:
            buffer_size: Number of frames to keep in memory (default: 10)
        """
        self.buffer_size = buffer_size
        self.age_buffer = deque(maxlen=buffer_size)
        self.gender_buffer = deque(maxlen=buffer_size)

    def update_and_get_average(self, new_age, new_gender):
        """
        Adds new predictions and returns smoothed results.
        
        Strategy:
        - Age: Uses most frequent from recent predictions
        - Gender: Requires 60%+ consensus to prevent flickering
        
        Args:
            new_age: New age prediction (string)
            new_gender: New gender prediction (string)
            
        Returns:
            Tuple of (final_age, final_gender)
        """
        self.age_buffer.append(new_age)
        self.gender_buffer.append(new_gender)
        
        # AGE SMOOTHING: Use most recent predictions with higher weight
        if len(self.age_buffer) >= 3:
            # Get last 3-5 predictions (most stable and recent)
            recent_count = min(5, len(self.age_buffer))
            recent_ages = list(self.age_buffer)[-recent_count:]
            
            # Find most frequent age in recent predictions
            final_age = max(set(recent_ages), key=recent_ages.count)
        else:
            # Not enough data yet, use current prediction
            final_age = new_age
        
        # GENDER SMOOTHING: Require strong consensus (reduces flickering)
        if len(self.gender_buffer) >= 5:
            male_count = self.gender_buffer.count('Male')
            female_count = self.gender_buffer.count('Female')
            
            total = male_count + female_count
            if total > 0:
                male_ratio = male_count / total
                female_ratio = female_count / total
                
                # Require 60% consensus before committing to a gender
                if male_ratio > 0.6:
                    final_gender = 'Male'
                elif female_ratio > 0.6:
                    final_gender = 'Female'
                else:
                    # No strong consensus, keep previous stable prediction
                    final_gender = list(self.gender_buffer)[-1]
            else:
                final_gender = new_gender
        else:
            # Not enough data yet, use current prediction
            final_gender = new_gender
            
        return final_age, final_gender

    def reset(self):
        """Clear all buffered predictions"""
        self.age_buffer.clear()
        self.gender_buffer.clear()

    def get_buffer_status(self):
        """
        Returns current buffer fill status for debugging.
        
        Returns:
            Dictionary with buffer statistics
        """
        return {
            'age_count': len(self.age_buffer),
            'gender_count': len(self.gender_buffer),
            'max_size': self.buffer_size
        }