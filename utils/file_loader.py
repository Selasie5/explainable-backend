import os
import pickle
import pandas as pd
import joblib
from sklearn.base import BaseEstimator
from fastapi import UploadFile, HTTPException, status
from typing import Union, Tuple

SUPPORTED_MODEL_TYPES=(
     "sklearn", "xgboost", "lightgbm"
)

def load_csV_files(file: UploadFile)->pd.DataFrame:
    try:
        return pd.read_csv(file.file)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read CSV file. Please ensure it's a vaild csv format"
        )
   

def load_model_file(file: UploadFile)->Union[BaseEstimator, object]:
    try:
        model = joblib.load(file.file)
    except Exception:
        try:
            file.file.seek(0)
            model = pickle.load(file.file)
        except Exception:
          raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail = "Failed to load model. Please make  sure it's a valid .pkl file "
          )
    return model


def validate_model_compatibility(model, data:pd.DataFrame):
    #Checks if the model has a .predict method and support s the input shape
    if not hasattr(model, "predict"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= "Uploaded model does not have a `predict` method."
        )
    try:
        #Make a  dry run prediction on the model
        model.predict(data.head(1))
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail= "Model is not compatible with the uploaded CSV data. Ensure the features match."
        )
