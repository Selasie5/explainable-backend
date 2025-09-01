import pandas as pd
from tensorflow import keras
import numpy as np

def test_keras_cli():
    X = pd.DataFrame({
        'feature_1': [1, 4, 7],
        'feature_2': [2, 5, 8],
        'feature_3': [3, 6, 9]
    })
    y = np.array([0, 1, 0])
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    model.fit(X.values, y, epochs=100, verbose=0)
    model.save('mock_model_keras.h5')
    X.to_csv('mock_data_keras.csv', index=False)
    # CLI command example (to run manually):
    # python cli_explain.py --model mock_model_keras.h5 --data mock_data_keras.csv --method shap
