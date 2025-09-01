import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

def test_sklearn_cli():
    # Create mock data
    X = pd.DataFrame({
        'feature_1': [1, 4, 7],
        'feature_2': [2, 5, 8],
        'feature_3': [3, 6, 9]
    })
    y = [0, 1, 0]
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, 'mock_model_sklearn.pkl')
    X.to_csv('mock_data_sklearn.csv', index=False)
    # CLI command example (to run manually):
    # python cli_explain.py --model mock_model_sklearn.pkl --data mock_data_sklearn.csv --method shap
