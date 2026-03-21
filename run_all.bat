call venv\Scripts\activate
pip install -r requirements.txt
python src/features/rfm_features.py
python src/features/drift_features.py
python src/models/segmentation.py
python src/models/churn_model.py
python src/models/channel_model.py
python src/explainability/shap_explainer.py
python src/pipelines/full_pipeline.py
start cmd /k "venv\Scripts\activate & streamlit run dashboard/app.py"
