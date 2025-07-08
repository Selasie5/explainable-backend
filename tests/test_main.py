import pytest
from fastapi.testclient import TestClient
import sys
import os


# Add the workspace directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from main import app


client = TestClient(app)
def test_main_app():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_analyse_endpoint(mock_csv_file, mock_model_file):
    files = {
        "csv": ("mock_data.csv", mock_csv_file, "text/csv"),
        "model": ("mock_model.pkl", mock_model_file, "application/octet-stream")
    }
    response = client.post("/api/v1/analyze", files=files)
    assert response.status_code == 200
    response_data = response.json()
    assert response_data["message"] == "SHAP explanations generated successfully."
    assert "shap_summary_plot" in response_data
    assert "shap_force_plot" in response_data
    assert "feature_importance" in response_data
    
@pytest.fixture
def mock_csv_file():
    """Fixture for mock CSV file"""
    return b"feature_1,feature_2,feature_3\n1,2,3\n4,5,6\n7,8,9"

@pytest.fixture
def mock_model_file():
    """Fixture for mock model file"""
    from sklearn.linear_model import LogisticRegression
    import joblib
    import io
    import numpy as np

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    model = LogisticRegression()
    model.fit(X, y)

    model_file = io.BytesIO()
    joblib.dump(model, model_file)
    model_file.seek(0)
    return model_file.read()
