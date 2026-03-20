import pandas as pd
import numpy as np
from src.features.feature_utils import load_data, save_data

def preprocess_transactions(df):
    """Clean transaction data for RFM."""
    # Filter valid rows
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0) & (df['CustomerID'].notna())].copy()
    
    # Exclude cancellations (InvoiceNo starts with 'C')
    # Convert to string to safely use startswith
    df['InvoiceNoStr'] = df['InvoiceNo'].astype(str)
    df = df[~df['InvoiceNoStr'].str.startswith('C')].copy()
    
    # Calculate revenue
    df['revenue'] = df['Quantity'] * df['UnitPrice']
    
    # Parse date
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    return df

def extract_rfm(df_clean, reference_date=None):
    """Extract RFM features from clean transaction data."""
    if reference_date is None:
        reference_date = df_clean['InvoiceDate'].max()
        
    rfm = df_clean.groupby('CustomerID').agg(
        recency=('InvoiceDate', lambda x: (reference_date - x.max()).days),
        frequency=('InvoiceNo', 'nunique'),
        monetary=('revenue', 'sum'),
        transaction_count=('InvoiceNo', 'count')
    ).reset_index()
    
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
    
    # Cap monetary outliers at 99th percentile
    monetary_99 = rfm['monetary'].quantile(0.99)
    rfm['monetary'] = np.where(rfm['monetary'] > monetary_99, monetary_99, rfm['monetary'])
    
    return rfm

if __name__ == "__main__":
    import yaml
    with open('configs/model_config.yaml', 'r') as f:
        paths = yaml.safe_load(f)['paths']
        
    print("Loading raw data...")
    raw_df = load_data(paths['data']['raw'] + "SegmentnDrift.csv")
    clean_df = preprocess_transactions(raw_df)
    
    print("Extracting RFM...")
    rfm_features = extract_rfm(clean_df)
    
    save_data(rfm_features, paths['data']['processed'] + "rfm_features.csv")
    print("RFM feature extraction completed.")
