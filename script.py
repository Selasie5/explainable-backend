# save_mock_model.py
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Train a simple model
model = LogisticRegression()
model.fit(X, y)

# Save the model correctly
joblib.dump(model, "mock_model.pkl")
print("Model saved to mock_model.pkl")
