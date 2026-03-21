import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
from src.features.feature_utils import load_data, save_data

class SegmentationModel:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
        self.feature_cols = ['recency', 'frequency', 'monetary']
        
    def fit(self, df):
        """Fit the KMeans model and scaler."""
        X = df[self.feature_cols].copy()
        
        # Log1p transform for skewness
        X['monetary'] = np.log1p(X['monetary'])
        X['frequency'] = np.log1p(X['frequency'])
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        
    def predict(self, df):
        """Predict segment IDs."""
        X = df[self.feature_cols].copy()
        X['monetary'] = np.log1p(X['monetary'])
        X['frequency'] = np.log1p(X['frequency'])
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def assign_segment_names(self, df):
        """Map cluster IDs to business names based on profiling."""
        # Calculate centroids dynamically
        centroids = df.groupby('segment_id')[self.feature_cols].mean()
        
        # Sort clusters by monetary value
        sorted_clusters = centroids.sort_values(by='monetary', ascending=False).index.tolist()
        
        # Simple mapping assuming 4 clusters
        mapping = {
            sorted_clusters[0]: 'premium_active',
            sorted_clusters[1]: 'high_value_declining',
            sorted_clusters[2]: 'at_risk_mid_tier',
            sorted_clusters[3]: 'dormant_low_value'
        }
        
        # If n_clusters is different, just return the ID as string
        if self.n_clusters != 4:
            mapping = {i: f'segment_{i}' for i in range(self.n_clusters)}
            
        df['segment_name'] = df['segment_id'].map(mapping)
        return df

    def save(self, model_path, scaler_path):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
            
    def load(self, model_path, scaler_path):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

if __name__ == "__main__":
    import yaml
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading RFM data...")
    rfm_df = load_data(config['paths']['data']['processed'] + "rfm_features.csv")
    
    print("Training Segmentation Model...")
    seg_model = SegmentationModel(
        n_clusters=config['segmentation']['n_clusters'],
        random_state=config['segmentation']['random_state']
    )
    
    seg_model.fit(rfm_df)
    rfm_df['segment_id'] = seg_model.predict(rfm_df)
    rfm_df = seg_model.assign_segment_names(rfm_df)
    
    print("Saving Segment model and outputs...")
    seg_model.save(
        config['paths']['artifacts']['folder'] + "kmeans_model.pkl",
        config['paths']['artifacts']['folder'] + "segment_scaler.pkl"
    )
    save_data(rfm_df[['CustomerID', 'segment_id', 'segment_name']], config['paths']['outputs']['folder'] + "segment_output.csv")
    print("Segmentation completed.")
