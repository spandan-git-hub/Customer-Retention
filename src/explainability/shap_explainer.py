import shap
import pandas as pd
import numpy as np
import joblib
from src.features.feature_utils import load_data, save_data
from src.models.churn_model import ChurnModel

class ShapExplainer:
    def __init__(self, model):
        # model is the trained lgb.Booster
        self.explainer = shap.TreeExplainer(model)
        
        # Simple tag mapping based on implementation plan
        self.tag_mapping = {
            'Balance': ('declining_balance', 'high_balance_risk'), # (negative shap tag, positive shap tag)
            'NumOfProducts': ('low_product_usage', 'high_product_risk'),
            'IsActiveMember': ('low_account_activity', 'active_but_leaving'),
            'spend_change_ratio': ('falling_spend_trend', 'volatile_spend'),
            'txn_drop_ratio': ('stable_transactions', 'transaction_frequency_drop'),
            'Age': ('young_flight_risk', 'tenure_aging_risk'),
            'CreditScore': ('credit_stress_signal', 'good_credit_leaving'),
            'recency_change': ('stable_recency', 'increasing_inactivity')
        }
        
    def generate_explanations(self, df_features, customer_ids):
        """Generate top 3 reasons per customer."""
        shap_values = self.explainer.shap_values(df_features)
        
        # lightgbm binary objective returns list of 2 or single array depending on version/setup.
        if isinstance(shap_values, list):
            sv = shap_values[1] # positive class
        else:
            sv = shap_values
            
        feature_names = df_features.columns.tolist()
        
        results = []
        for i in range(len(sv)):
            # get top 3 by absolute magnitude
            instance_shap = sv[i]
            top_indices = np.argsort(np.abs(instance_shap))[-3:][::-1]
            
            reasons = []
            for idx in top_indices:
                feat = feature_names[idx]
                val = instance_shap[idx]
                
                # Use mapping, if feature not in mapping, just use feature name
                if feat in self.tag_mapping:
                    tag = self.tag_mapping[feat][0] if val < 0 else self.tag_mapping[feat][1]
                else:
                     tag = f"{feat}_issue"
                     
                reasons.append(tag)
                
            summary = f"Risk driven by {reasons[0].replace('_', ' ')}, {reasons[1].replace('_', ' ')}, and {reasons[2].replace('_', ' ')}."
            
            results.append({
                'customer_id': customer_ids.iloc[i],
                'top_reason_1': reasons[0] if len(reasons) > 0 else None,
                'top_reason_2': reasons[1] if len(reasons) > 1 else None,
                'top_reason_3': reasons[2] if len(reasons) > 2 else None,
                'explanation_summary': summary
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    import yaml
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading Churn data for SHAP explanations...")
    raw_df = load_data(config['paths']['data']['raw'] + "ChurnPrediction.csv")
    
    churn_model = ChurnModel()
    churn_model.load(
        config['paths']['artifacts']['folder'] + "churn_model.pkl",
        config['paths']['artifacts']['folder'] + "churn_encoders.pkl"
    )
    
    # Needs to match exactly what went into fit
    df_features = churn_model.preprocess(raw_df, is_train=False)
    
    print("Generating SHAP explanations...")
    explainer = ShapExplainer(churn_model.model)
    explanations_df = explainer.generate_explanations(df_features, raw_df['CustomerId'])
    
    print("Saving explanations...")
    save_data(explanations_df, config['paths']['outputs']['folder'] + "explanations.csv")
    print("SHAP explanations completed.")
