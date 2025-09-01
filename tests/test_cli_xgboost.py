import joblib
import pandas as pd
from xgboost import XGBClassifier

def test_xgboost_cli():
    X = pd.DataFrame({
        'feature_1': [1, 4, 7],
        'feature_2': [2, 5, 8],
        'feature_3': [3, 6, 9]
    })
    y = [0, 1, 0]
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    model.save_model('mock_model_xgboost.json')
    X.to_csv('mock_data_xgboost.csv', index=False)
    # CLI command example (to run manually):
    # python cli_explain.py --model mock_model_xgboost.json --data mock_data_xgboost.csv --method shap
