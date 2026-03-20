import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from src.features.feature_utils import load_data, save_data

class ChurnModel:
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'verbose': -1
            }
        else:
            self.params = params
            self.params['objective'] = 'binary'
            self.params['metric'] = 'auc'
            self.params['verbose'] = -1
            
        self.model = None
        self.encoders = {}
        
    def engineer_features(self, df):
        """Feature engineering specific to churn model."""
        df_out = df.copy()
        df_out['balance_per_product'] = df_out['Balance'] / (df_out['NumOfProducts'] + 1)
        df_out['salary_to_balance_ratio'] = df_out['EstimatedSalary'] / (df_out['Balance'] + 1)
        df_out['is_young_inactive'] = ((df_out['Age'] < 30) & (df_out['IsActiveMember'] == 0)).astype(int)
        
        # Provide default drift and segment features if missing (conceptually mock)
        if 'segment_name' not in df_out.columns:
            # simple mock rule for demo since they are diff datasets
            df_out['segment_name'] = np.where(df_out['Balance'] > 100000, 'premium_active', 'dormant_low_value')
            
        if 'spend_change_ratio' not in df_out.columns:
            df_out['spend_change_ratio'] = 0.0
            df_out['txn_drop_ratio'] = 0.0
            df_out['recency_change'] = 0.0
            
        return df_out

    def preprocess(self, df, is_train=True):
        X = self.engineer_features(df)
        
        # Hardcoded features matching feature_config
        self.cat_cols = ['Geography', 'Gender', 'segment_name']
        self.num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                    'EstimatedSalary', 'balance_per_product', 'salary_to_balance_ratio',
                    'spend_change_ratio', 'txn_drop_ratio', 'recency_change',
                    'HasCrCard', 'IsActiveMember', 'is_young_inactive']
        
        features = self.num_cols + self.cat_cols
        X = X[features]
        
        # Encoding categorical variables
        for c in self.cat_cols:
            if is_train:
                le = LabelEncoder()
                X[c] = le.fit_transform(X[c].astype(str))
                self.encoders[c] = le
            else:
                le = self.encoders[c]
                # handle unseen labels
                X[c] = X[c].map(lambda s: s if s in le.classes_ else le.classes_[0])
                X[c] = le.transform(X[c].astype(str))
        
        return X

    def fit(self, df, target_col='Exited'):
        X = self.preprocess(df, is_train=True)
        y = df[target_col]
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=self.cat_cols)
        
        # early_stopping in params or callbacks
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        return self

    def predict_proba(self, df):
        X = self.preprocess(df, is_train=False)
        return self.model.predict(X)

    def attach_risk_bands(self, df_scores, thresholds):
        df_scores['risk_band'] = pd.cut(
            df_scores['churn_score'], 
            bins=[-np.inf, thresholds['low_risk'], thresholds['medium_risk'], thresholds['high_risk'], np.inf],
            labels=['low_risk', 'medium_risk', 'high_risk', 'critical_risk']
        )
        return df_scores

    def save(self, model_path, encoder_path):
        import joblib
        joblib.dump(self.model, model_path)
        joblib.dump(self.encoders, encoder_path)

    def load(self, model_path, encoder_path):
        import joblib
        self.model = joblib.load(model_path)
        self.encoders = joblib.load(encoder_path)

if __name__ == "__main__":
    import yaml
    with open('configs/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    print("Loading Churn data...")
    raw_df = load_data(config['paths']['data']['raw'] + "ChurnPrediction.csv")
    
    print("Training Churn Model...")
    churn_model = ChurnModel(params=config['churn_model']['params'])
    churn_model.fit(raw_df)
    
    # Predict on training data just for output generation
    scores = churn_model.predict_proba(raw_df)
    
    output_df = pd.DataFrame({
        'CustomerId': raw_df['CustomerId'],
        'churn_score': scores
    })
    
    output_df = churn_model.attach_risk_bands(output_df, config['churn_model']['thresholds'])
    
    print("Saving Churn model and outputs...")
    churn_model.save(
        config['paths']['artifacts']['folder'] + "churn_model.pkl",
        config['paths']['artifacts']['folder'] + "churn_encoders.pkl"
    )
    save_data(output_df, config['paths']['outputs']['folder'] + "churn_scores.csv")
    print("Churn Model pipeline completed.")
