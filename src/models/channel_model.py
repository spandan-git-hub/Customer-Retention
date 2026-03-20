import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from src.features.feature_utils import load_data, save_data

class ChannelModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_leaf': 2,
                'random_state': 42,
                'class_weight': 'balanced'
            }
        else:
            self.params = params
            
        self.model = RandomForestClassifier(**self.params)
        self.encoders = {}
        self.cat_cols = ['job', 'marital', 'education', 'poutcome', 'segment_name_encoded']
        self.num_cols = ['age', 'balance', 'campaign', 'previous', 'pdays', 'churn_score']
        
    def engineer_features(self, df):
        df_out = df.copy()
        
        # Ensure pdays=-1 is handled logically (e.g. 999 days)
        if 'pdays' in df_out.columns:
            df_out['pdays'] = np.where(df_out['pdays'] == -1, 999, df_out['pdays'])
            
        # Enrich with conceptual values if testing standalone
        if 'segment_name_encoded' not in df_out.columns:
            df_out['segment_name_encoded'] = 'default'
        if 'churn_score' not in df_out.columns:
            df_out['churn_score'] = 0.5
            
        # Map target (y) and channel (contact) into best_channel
        # If y=yes, target = contact, else we might drop or assign a fallback
        if 'y' in df_out.columns and 'contact' in df_out.columns:
            df_out['best_channel'] = np.where(
                df_out['contact'] == 'cellular', 'SMS',
                np.where(df_out['contact'] == 'telephone', 'phone_call', 'email')
            )
            # Only train on successful campaigns or treat as multi-class target directly
            # For simplicity, we predict what channel was used given the profile
            
        return df_out

    def preprocess(self, df, is_train=True):
        X = self.engineer_features(df)
        
        # Keep only available columns
        available_cat = [c for c in self.cat_cols if c in X.columns]
        available_num = [c for c in self.num_cols if c in X.columns]
        
        features = available_num + available_cat
        X_feats = X[features].copy()
        
        for c in available_cat:
            if is_train:
                le = LabelEncoder()
                X_feats[c] = le.fit_transform(X_feats[c].astype(str))
                self.encoders[c] = le
            else:
                le = self.encoders.get(c, None)
                if le is not None:
                    # handle unseen using the 0th class
                    class_set = set(le.classes_)
                    X_feats[c] = X_feats[c].map(lambda s: s if s in class_set else le.classes_[0])
                    X_feats[c] = le.transform(X_feats[c].astype(str))
                else:
                    X_feats[c] = 0
                    
        return X_feats

    def fit(self, df):
        df_eng = self.engineer_features(df)
        
        # For training, only use rows where campaign was successful if available
        if 'y' in df_eng.columns:
            df_train = df_eng[df_eng['y'] == 'yes']
            if len(df_train) == 0:
                df_train = df_eng # Fallback
        else:
            df_train = df_eng
            
        X = self.preprocess(df_train, is_train=True)
        y = df_train['best_channel']
        
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, df):
        X = self.preprocess(df, is_train=False)
        preds = self.model.predict(X)
        probs = self.model.predict_proba(X)
        confidence = np.max(probs, axis=1)
        return preds, confidence

    def save(self, model_path, encoder_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.encoders, encoder_path)

    def load(self, model_path, encoder_path):
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoder_path)

if __name__ == "__main__":
    import yaml
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading Channel data...")
    raw_df = load_data(config['paths']['data']['raw'] + "ChannelPrediction.csv")
    
    print("Training Channel Model...")
    channel_model = ChannelModel(params=config['channel_model']['params'])
    channel_model.fit(raw_df)
    
    preds, confs = channel_model.predict(raw_df)
    
    output_df = pd.DataFrame({
        # Since no CustomerID in this dataset, using index
        'customer_id': raw_df.index,
        'best_channel': preds,
        'channel_confidence': confs
    })
    
    print("Saving Channel model and outputs...")
    channel_model.save(
        config['paths']['artifacts']['folder'] + "channel_model.pkl",
        config['paths']['artifacts']['folder'] + "channel_encoders.pkl"
    )
    save_data(output_df, config['paths']['outputs']['folder'] + "channel_recommendations.csv")
    print("Channel Model pipeline completed.")
