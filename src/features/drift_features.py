import pandas as pd
import numpy as np
from src.features.feature_utils import load_data, save_data
from src.features.rfm_features import preprocess_transactions

def extract_drift(df_clean, reference_date=None):
    """Extract rolling time-window feature drift from transaction data."""
    if reference_date is None:
        reference_date = df_clean['InvoiceDate'].max()
        
    df_clean['days_ago'] = (reference_date - df_clean['InvoiceDate']).dt.days

    # We will build features customer by customer
    # Using groupby and aggregations
    results = []
    
    # Fast vectorized approach across groups
    # Spend arrays
    spend_30 = df_clean[df_clean['days_ago'] <= 30].groupby('CustomerID')['revenue'].sum()
    spend_60_all = df_clean[df_clean['days_ago'] <= 60].groupby('CustomerID')['revenue'].sum()
    spend_90 = df_clean[df_clean['days_ago'] <= 90].groupby('CustomerID')['revenue'].sum()
    
    txn_30 = df_clean[df_clean['days_ago'] <= 30].groupby('CustomerID')['InvoiceNo'].nunique()
    txn_90 = df_clean[df_clean['days_ago'] <= 90].groupby('CustomerID')['InvoiceNo'].nunique()

    # Recency helpers
    rec_30 = df_clean[df_clean['days_ago'] <= 30].groupby('CustomerID').agg(
        last_in_30=('days_ago', 'min')
    )
    rec_90 = df_clean[df_clean['days_ago'] <= 90].groupby('CustomerID').agg(
        last_in_90=('days_ago', 'min')
    )
    
    # Active months
    df_clean['month_yr'] = df_clean['InvoiceDate'].dt.to_period('M')
    active_months = df_clean.groupby('CustomerID')['month_yr'].nunique()

    # Construct complete dataframe
    all_customers = df_clean[['CustomerID']].drop_duplicates().set_index('CustomerID')
    
    drift_df = pd.DataFrame(index=all_customers.index)
    
    # Assign and impute zeros where no activity in that window
    drift_df['spend_last_30d'] = spend_30.reindex(drift_df.index).fillna(0)
    
    spend_60_cum = spend_60_all.reindex(drift_df.index).fillna(0)
    spend_60_only = spend_60_cum - drift_df['spend_last_30d']
    
    drift_df['spend_last_90d'] = spend_90.reindex(drift_df.index).fillna(0)
    
    t_30 = txn_30.reindex(drift_df.index).fillna(0)
    t_90 = txn_90.reindex(drift_df.index).fillna(0)
    drift_df['txn_count_last_30d'] = t_30
    
    drift_df['spend_change_ratio'] = (drift_df['spend_last_30d'] - spend_60_only) / (spend_60_only + 1)
    drift_df['txn_drop_ratio'] = (t_90 - (t_30 * 3)) / (t_90 + 1) # Normalized comparing 90d to 3x 30d
    
    # Recency change
    # If no txn in 30d, recency is >30, compare against 90d recency
    r_30 = rec_30['last_in_30'].reindex(drift_df.index).fillna(31) # assuming no activity means >=31
    r_90 = rec_90['last_in_90'].reindex(drift_df.index).fillna(91)
    drift_df['recency_change'] = r_30 - r_90
    
    drift_df['active_months'] = active_months.reindex(drift_df.index).fillna(0)
    
    return drift_df.reset_index()

if __name__ == "__main__":
    import yaml
    with open('configs/model_config.yaml', 'r') as f:
        paths = yaml.safe_load(f)['paths']
        
    print("Loading raw data...")
    raw_df = load_data(paths['data']['raw'] + "SegmentnDrift.csv")
    clean_df = preprocess_transactions(raw_df)
    
    print("Extracting drift features...")
    drift_features = extract_drift(clean_df)
    
    save_data(drift_features, paths['data']['processed'] + "drift_features.csv")
    print("Drift feature extraction completed.")
