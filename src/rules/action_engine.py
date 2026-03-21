import pandas as pd
import numpy as np
from src.features.feature_utils import load_data, save_data
import yaml

class ActionEngine:
    def __init__(self, config_path='configs/rules_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.priorities = self.config['action_priority']
        self.fallback = self.config['channel_fallback']
        
    def determine_action(self, row):
        """Map risk_band, segment, reason, and channel to a specific action."""
        # row expects: churn_score, risk_band, segment_name, top_reason_1, best_channel
        
        rb = row.get('risk_band', 'low_risk')
        seg = row.get('segment_name', 'default')
        reason = row.get('top_reason_1', '')
        channel = row.get('best_channel', self.fallback.get(seg, self.fallback['default']))
        
        # Decision Table Logic implementation
        action = 'MONITOR_ONLY'
        note = 'Hold action — monitor for 30 days'
        
        if rb == 'critical_risk':
            if seg == 'premium_active':
                action = 'RM_CALLBACK'
                note = 'Immediate RM callback + escalation'
                channel = 'phone_call'
            elif seg == 'high_value_declining' and reason == 'declining_balance':
                action = 'RM_VISIT'
                note = 'Service recovery + personalized retention offer'
            elif reason == 'low_account_activity':
                action = 'SERVICE_RECOVERY'
                note = 'Reactivation call + product discovery'
                channel = 'phone_call'
            else:
                action = 'RM_CALLBACK'
                note = 'Critical risk catch-all escalation'
                
        elif rb == 'high_risk':
            if seg == 'high_value_declining' and reason == 'transaction_frequency_drop':
                action = 'LOYALTY_OFFER'
                note = 'Loyalty reward offer'
                channel = 'phone_call'
            elif reason == 'credit_stress_signal':
                action = 'FEE_WAIVER'
                note = 'Financial counseling nudge + fee waiver offer'
                channel = 'SMS'
            else:
                action = 'PRODUCT_TUTORIAL'
                note = 'Product tutorial + digital engagement prompt'
                channel = 'push_notification'
                
        elif rb == 'medium_risk':
            if reason == 'falling_spend_trend':
                action = 'DIGITAL_PROMPT'
                note = 'Targeted engagement offer'
                channel = 'SMS'
            elif seg == 'dormant_low_value':
                action = 'EMAIL_NUDGE'
                note = 'Low-cost reactivation email nudge'
                channel = 'email'
            else:
                action = 'EMAIL_NUDGE'
                note = 'General medium risk nudge'
                
        else: # low risk
            action = 'MONITOR_ONLY'
            note = 'Monitor only — newsletter / educational content'
            channel = 'email'

        if 'false_positive' in str(reason).lower():
             action = 'HOLD_ACTION'
             note = 'False positive suspected'
             
        priority = self.priorities.get(action, 99)
        
        return pd.Series([action, priority, note, channel])

    def run(self, df_merged):
        """Run engine on merged dataframe of predictions and explanations."""
        cols = ['chosen_action', 'action_priority', 'action_note', 'execution_channel']
        df_merged[cols] = df_merged.apply(self.determine_action, axis=1)
        return df_merged

if __name__ == "__main__":
    print("Action Engine can be tested via full_pipeline.py merging outputs.")
