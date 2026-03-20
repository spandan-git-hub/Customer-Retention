import pandas as pd
import numpy as np
import yaml
from pathlib import Path

# Importers for our modular pipeline
from src.features.feature_utils import load_data, save_data
from src.features.rfm_features import extract_rfm, preprocess_transactions
from src.features.drift_features import extract_drift
from src.models.segmentation import SegmentationModel
from src.models.churn_model import ChurnModel
from src.explainability.shap_explainer import ShapExplainer
from src.models.channel_model import ChannelModel
from src.rules.action_engine import ActionEngine

def run_full_pipeline():
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("================== PHASE 1: DATA COMPONENT ==================")
    print("Loading raw datasets...")
    # Normally we load Segments, Churn, Channel data. 
    # For a unified view on the demo, we use the churn dataset as the base 
    # for the customers we want to reach.
    churn_raw = load_data(config['paths']['data']['raw'] + "ChurnPrediction.csv")
    
    print("================== PHASE 2 & 3: PREDICTION ENGINE ==================")
    # Load and prep Churn Model
    churn_model = ChurnModel()
    churn_model.load(
        config['paths']['artifacts']['folder'] + "churn_model.pkl",
        config['paths']['artifacts']['folder'] + "churn_encoders.pkl"
    )
    
    # Generate scores
    scores = churn_model.predict_proba(churn_raw)
    df_results = pd.DataFrame({
        'customer_id': churn_raw['CustomerId'],
        'churn_score': scores
    })
    
    # Assign Risk Bands
    df_results = churn_model.attach_risk_bands(df_results, config['churn_model']['thresholds'])
    
    # In a full production system, we'd join segments and drift here by real CustomerID
    # For MVP joined simulation, we mock the logic that was defined in feature_engineering:
    df_results['segment_name'] = np.where(churn_raw['Balance'] > 100000, 'premium_active', 
                                np.where(churn_raw['Balance'] == 0, 'dormant_low_value', 'at_risk_mid_tier'))
                                
    print("================== PHASE 4: EXPLAINABILITY ==================")
    df_features = churn_model.preprocess(churn_raw, is_train=False)
    explainer = ShapExplainer(churn_model.model)
    explanations_df = explainer.generate_explanations(df_features, churn_raw['CustomerId'])
    
    # Merge Explanations
    df_results = df_results.merge(explanations_df, on='customer_id', how='left')
    
    print("================== PHASE 5: CHANNEL OPTIMIZATION ==================")
    # Load Channel Model
    channel_model = ChannelModel()
    channel_model.load(
        config['paths']['artifacts']['folder'] + "channel_model.pkl",
        config['paths']['artifacts']['folder'] + "channel_encoders.pkl"
    )
    
    # We prepare a mock dataframe based on the churn population to score channels
    # The channel model used 'age', 'balance', 'segment_name_encoded', 'churn_score' ...
    channel_input = churn_raw.copy()
    channel_input['churn_score'] = scores
    channel_input['segment_name_encoded'] = df_results['segment_name']
    
    preds, confs = channel_model.predict(channel_input)
    df_results['best_channel'] = preds
    df_results['channel_confidence'] = confs
    
    print("================== PHASE 6: NEXT BEST ACTION ==================")
    action_engine = ActionEngine()
    
    # We need to map dataframe format to the action engine apply logic
    # The rule engine reads from each row
    action_df = df_results.copy()
    
    # Fix apply logic structure from Rules engine
    def apply_action(row):
        return action_engine.determine_action(row)
        
    action_outputs = action_df.apply(apply_action, axis=1)
    action_outputs.columns = ['chosen_action', 'action_priority', 'action_note', 'execution_channel']
    
    df_results = pd.concat([df_results, action_outputs], axis=1)
    
    print("================== PHASE 7: OUTPUT & LOGGING ==================")
    # Save the master table
    master_path = config['paths']['outputs']['folder'] + "master_action_table.csv"
    save_data(df_results, master_path)
    
    # Generate mock outcome log based on the run
    outcome_log = df_results[['customer_id', 'churn_score', 'risk_band', 'segment_name', 
                              'top_reason_1', 'execution_channel', 'chosen_action']].copy()
    outcome_log.rename(columns={'top_reason_1': 'predicted_reason'}, inplace=True)
    outcome_log['timestamp'] = pd.Timestamp.now()
    outcome_log['outreach_sent'] = outcome_log['chosen_action'] != 'MONITOR_ONLY'
    outcome_log['responded'] = outcome_log['outreach_sent'] & (np.random.rand(len(outcome_log)) > 0.6)
    outcome_log['retained_90d'] = outcome_log['responded'] | (outcome_log['risk_band'] == 'low_risk')
    
    save_data(outcome_log, config['paths']['outputs']['folder'] + "outreach_outcomes.csv")
    
    print("Retention Decision Engine pipeline completed successfully!")
    print(df_results.head())

if __name__ == "__main__":
    run_full_pipeline()
