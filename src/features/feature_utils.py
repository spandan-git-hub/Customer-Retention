import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath, **kwargs):
    """Load CSV data with error handling, encoding fallback, and separator detection."""
    try:
        df = pd.read_csv(filepath, **kwargs)
        if df.shape[1] <= 1: # Possibly wrong separator
             df_tab = pd.read_csv(filepath, sep='\t', **kwargs)
             if df_tab.shape[1] > df.shape[1]:
                 df = df_tab
                 logging.info(f"Loaded data from {filepath} with shape {df.shape} using tab separator.")
             else:
                 logging.info(f"Loaded data from {filepath} with shape {df.shape}")
        else:
            logging.info(f"Loaded data from {filepath} with shape {df.shape}")
        return df
    except UnicodeDecodeError:
        logging.warning(f"UTF-8 decoding failed for {filepath}. Retrying with ISO-8859-1.")
        try:
            # Try with default then tab
            df = pd.read_csv(filepath, encoding='ISO-8859-1', **kwargs)
            if df.shape[1] <= 1:
                df_tab = pd.read_csv(filepath, sep='\t', encoding='ISO-8859-1', **kwargs)
                if df_tab.shape[1] > df.shape[1]:
                    df = df_tab
            return df
        except Exception as e:
            logging.error(f"Error loading {filepath} with ISO-8859-1: {str(e)}")
            raise
    except Exception as e:
        logging.error(f"Error loading {filepath}: {str(e)}")
        raise

def save_data(df, filepath, index=False):
    """Save dataframe to CSV."""
    df.to_csv(filepath, index=index)
    logging.info(f"Saved data to {filepath} with shape {df.shape}")
