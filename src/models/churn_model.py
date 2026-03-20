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
        
        # Map columns from Credit Card Customers dataset if present
        mapping = {
            'Customer_Age': 'Age',
            'Total_Revolving_Bal': 'Balance',
            'Total_Relationship_Count': 'NumOfProducts',
            'Income_Category': 'EstimatedSalary',
            'Months_Inactive_12_mon': 'IsActiveMember', # Proxy
            'Attrition_Flag': 'Exited',
            'year_birth': 'Age_Birth_Proxy', # if available
            'balance': 'Balance',
            'age': 'Age'
        }
        for old, new in mapping.items():
            if old in df_out.columns and new not in df_out.columns:
                df_out[new] = df_out[old]
                
        # Handle target encoding
        if 'Exited' in df_out.columns and df_out['Exited'].dtype == object:
            df_out['Exited'] = (df_out['Exited'] == 'Attrited Customer').astype(int)
            
        # Ensure numeric types
        for col in ['Balance', 'NumOfProducts', 'Age']:
            if col in df_out.columns:
                df_out[col] = pd.to_numeric(df_out[col], errors='coerce').fillna(0)

        df_out['balance_per_product'] = df_out['Balance'] / (df_out['NumOfProducts'] + 1)
        df_out['salary_to_balance_ratio'] = 0.0 # difficult to map from categories easily
        df_out['is_young_inactive'] = ((df_out['Age'] < 30)).astype(int) # proxy
        
        if 'segment_name' not in df_out.columns:
            df_out['segment_name'] = 'premium_active'
            
        if 'spend_change_ratio' not in df_out.columns:
            df_out['spend_change_ratio'] = 0.0
            df_out['txn_drop_ratio'] = 0.0
            df_out['recency_change'] = 0.0
            
        return df_out

    def preprocess(self, df, is_train=True):
        X = self.engineer_features(df)
        
        # Fallback for categorical columns
        for c in ['Geography', 'Gender', 'segment_name', 'sim_txn_trend', 'sim_mobile_app_usage', 'sim_income_change']:
            if c not in X.columns:
                X[c] = 'Unknown'
        
        # Fallback for numerical columns
        for c in ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                  'EstimatedSalary', 'balance_per_product', 'salary_to_balance_ratio',
                  'spend_change_ratio', 'txn_drop_ratio', 'recency_change',
                  'HasCrCard', 'IsActiveMember', 'is_young_inactive',
                  'sim_days_since_last_txn', 'sim_monthly_avg_txns', 'sim_category_pref_score',
                  'sim_product_diversity_index', 'sim_session_freq_score', 'sim_online_offline_ratio',
                  'sim_complaint_count_12m', 'sim_complaint_severity', 'sim_resolution_time_days',
                  'sim_relocation_flag', 'sim_job_change_prob', 'sim_price_sensitivity',
                  'sim_competitor_interaction', 'sim_offer_responsiveness']:
            if c not in X.columns:
                X[c] = 0.0
            else:
                X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

        self.cat_cols = ['Geography', 'Gender', 'segment_name', 'sim_txn_trend', 'sim_mobile_app_usage', 'sim_income_change']
        self.num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
                    'EstimatedSalary', 'balance_per_product', 'salary_to_balance_ratio',
                    'spend_change_ratio', 'txn_drop_ratio', 'recency_change',
                    'HasCrCard', 'IsActiveMember', 'is_young_inactive',
                    'sim_days_since_last_txn', 'sim_monthly_avg_txns', 'sim_category_pref_score',
                    'sim_product_diversity_index', 'sim_session_freq_score', 'sim_online_offline_ratio',
                    'sim_complaint_count_12m', 'sim_complaint_severity', 'sim_resolution_time_days',
                    'sim_relocation_flag', 'sim_job_change_prob', 'sim_price_sensitivity',
                    'sim_competitor_interaction', 'sim_offer_responsiveness']
        
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
        # We need the preprocessed df for the target as well if it was mapped
        df_mapped = self.engineer_features(df)
        X = self.preprocess(df, is_train=True)
        y = df_mapped[target_col]
        
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
        'CustomerId': raw_df['CLIENTNUM'] if 'CLIENTNUM' in raw_df.columns else raw_df.index,
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
