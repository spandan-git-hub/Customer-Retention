import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath, **kwargs):
    """Load CSV data with error handling."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        logging.info(f"Loaded data from {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading {filepath}: {str(e)}")
        raise

def save_data(df, filepath, index=False):
    """Save dataframe to CSV."""
    df.to_csv(filepath, index=index)
    logging.info(f"Saved data to {filepath} with shape {df.shape}")
