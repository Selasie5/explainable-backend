import joblib
import pandas as pd
from lightgbm import LGBMClassifier

def test_lightgbm_cli():
    X = pd.DataFrame({
        'feature_1': [1, 4, 7],
        'feature_2': [2, 5, 8],
        'feature_3': [3, 6, 9]
    })
    y = [0, 1, 0]
    model = LGBMClassifier()
    model.fit(X, y)
    model.booster_.save_model('mock_model_lightgbm.txt')
    X.to_csv('mock_data_lightgbm.csv', index=False)
    # CLI command example (to run manually):
    # python cli_explain.py --model mock_model_lightgbm.txt --data mock_data_lightgbm.csv --method shap
