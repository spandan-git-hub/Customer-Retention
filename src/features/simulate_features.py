import pandas as pd
import numpy as np
import os
import yaml
from pathlib import Path

def generate_behavioral_features(df, dataset_name):
    """
    Simulates realistic behavioral features for the dataset.
    Follows logical correlations:
    - High complaints -> higher churn 
    - High digital engagement -> lower churn
    - High price sensitivity -> higher competitor interaction
    """
    np.random.seed(42)
    n = len(df)
    
    # Extract any existing target correlations if they exist
    churn_prob_base = np.random.uniform(0.1, 0.3, n)
    if 'Exited' in df.columns:
        # Bank churn dataset
        churn_prob_base = np.where(df['Exited'] == 1, np.random.uniform(0.7, 1.0, n), np.random.uniform(0.0, 0.4, n))
    elif 'y' in df.columns:
        # Channel/Marketing dataset
        churn_prob_base = np.where(df['y'] == 'yes', np.random.uniform(0.0, 0.3, n), np.random.uniform(0.3, 0.6, n))
    
    # 1. Transaction Recency (enhanced)
    # Days since last transaction (0 to 365)
    # Correlated with churn - higher recency means higher churn likelihood
    recency_base = np.clip(churn_prob_base * 400 + np.random.normal(0, 30, n), 0, 365).astype(int)
    df['sim_days_since_last_txn'] = recency_base
    
    # 2. Transaction Frequency (normalized)
    # Monthly average transactions (0 to 50)
    freq_base = np.clip((1 - churn_prob_base) * 30 + np.random.normal(5, 10, n), 0, 50).astype(int)
    df['sim_monthly_avg_txns'] = freq_base
    
    trends = ['increasing', 'stable', 'decreasing']
    # If churn prob is high, trend is likely decreasing
    trend_probs = np.vstack([
        np.clip(0.8 - churn_prob_base, 0, 1), # increasing
        np.ones(n) * 0.4,                     # stable
        np.clip(churn_prob_base * 1.5, 0, 1)  # decreasing
    ])
    trend_probs = trend_probs / trend_probs.sum(axis=0) # normalize
    
    # Random choice with probabilities
    # Vectorized choice approximation:
    rands = np.random.rand(n)
    trend_idx = np.where(rands < trend_probs[0], 0, np.where(rands < trend_probs[0]+trend_probs[1], 1, 2))
    df['sim_txn_trend'] = [trends[i] for i in trend_idx]
    
    # 3. Product Usage Trends
    df['sim_category_pref_score'] = np.clip(np.random.normal(0.5, 0.2, n), 0.0, 1.0)
    df['sim_product_diversity_index'] = np.clip((1 - churn_prob_base) + np.random.normal(0, 0.2, n), 0.1, 1.0)
    
    # 4. Digital Engagement Patterns
    # Session frequency score (0-100), higher for low churn
    session_freq = np.clip((1 - churn_prob_base) * 100 + np.random.normal(0, 15, n), 0, 100).astype(int)
    df['sim_session_freq_score'] = session_freq
    
    app_usage_mapping = []
    for score in session_freq:
        if score < 30: app_usage_mapping.append('low')
        elif score < 70: app_usage_mapping.append('medium')
        else: app_usage_mapping.append('high')
    df['sim_mobile_app_usage'] = app_usage_mapping
    
    # Online vs offline transaction ratio (0.0 to 1.0)
    df['sim_online_offline_ratio'] = np.clip(np.random.normal(0.6, 0.2, n) + (session_freq/200), 0.0, 1.0)
    
    # 5. Complaint History (expanded)
    # Higher for higher churn
    complaint_count = np.clip(churn_prob_base * 10 + np.random.normal(0, 2, n), 0, 12).astype(int)
    df['sim_complaint_count_12m'] = complaint_count
    df['sim_complaint_severity'] = np.where(complaint_count > 0, np.clip(churn_prob_base + np.random.normal(0, 0.2, n), 0.1, 1.0), 0.0)
    df['sim_resolution_time_days'] = np.where(complaint_count > 0, np.clip(churn_prob_base * 30 + np.random.normal(5, 3, n), 1, 45).astype(int), 0)
    
    # 6. Life Events (simulated but realistic)
    income_changes = ['decrease', 'stable', 'increase']
    inc_idx = np.random.choice([0, 1, 2], size=n, p=[0.15, 0.7, 0.15])
    # Tweak: higher churn prob slightly increases chance of 'decrease'
    inc_idx = np.where((churn_prob_base > 0.7) & (np.random.rand(n) < 0.3), 0, inc_idx)
    df['sim_income_change'] = [income_changes[i] for i in inc_idx]
    
    df['sim_relocation_flag'] = np.where(np.random.rand(n) < 0.05, 1, 0)
    df['sim_job_change_prob'] = np.clip(np.random.beta(2, 8, n) + df['sim_relocation_flag']*0.3, 0.0, 1.0)
    
    # 7. Competitive Market Signals
    # Price sensitivity
    df['sim_price_sensitivity'] = np.clip(np.random.normal(0.5, 0.2, n), 0.0, 1.0)
    
    # High price sensitivity -> higher competitor interaction
    df['sim_competitor_interaction'] = np.clip(df['sim_price_sensitivity'] * 0.7 + churn_prob_base * 0.5 + np.random.normal(0, 0.1, n), 0.0, 1.0)
    
    # Offer responsiveness (higher for non-churn, price sensitive)
    df['sim_offer_responsiveness'] = np.clip((1 - churn_prob_base)*0.6 + df['sim_price_sensitivity']*0.4 + np.random.normal(0, 0.1, n), 0.0, 1.0)
    
    return df

def main():
    raw_dir = Path("d:/Interview, Internship & Stack/IDEA 2.0/Customer Retention/data/raw")
    source_dir = Path("d:/Interview, Internship & Stack/IDEA 2.0/Customer Retention/Datasets")
    
    # Ensure processed dir exists for outputs
    processed_dir = Path("d:/Interview, Internship & Stack/IDEA 2.0/Customer Retention/data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    files = ["ChannelPrediction.csv", "ChurnPrediction.csv", "SegmentnDrift.csv"]
    
    for f in files:
        source_path = source_dir / f
        dest_path = raw_dir / f
        
        print(f"Processing {f}...")
        try:
            # For massive datasets, doing it in chunks or reading specific columns might be needed,
            # but SegmentnDrift is 44MB, safely fits in memory with Pandas standard.
            df = pd.read_csv(source_path)
            
            # Add synthetic features
            df_enhanced = generate_behavioral_features(df, f)
            
            # Save into data/raw/ to act as the new gold standard "raw" input for the pipeline
            df_enhanced.to_csv(dest_path, index=False)
            print(f"Successfully generated 15 simulation columns for {f} -> {dest_path}")
            
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    main()
