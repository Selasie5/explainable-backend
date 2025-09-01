import os
from pyexpat import model
import joblib
import pickle

def load_sklearn_model(file_obj):
  file_obj.seek(0)
  try:
    return joblib.load(file_obj)
  except Exception as e:
    file_obj.seek(0)
    return pickle.load(file_obj)


def load_xgboost_model(file_obj):
  from xgboost import XGBClassifier
  import tempfile
  with tempfile.NamedTemporaryFile(delete= False, suffix=".json") as tmp:
    tmp.write(file_obj.read())
    tmp.flush()
    model = XGBClassifier()
    model.load_model(tmp.name)
  return model


def load_lightgbm_model(file_obj):
  from lightgbm import LGBMClassifier
  import tempfile
  with tempfile.NamedTemporaryFile(delete = False, suffix=".txt") as tmp:
    tmp.write(file_obj.read())
    tmp.flush()
    model = LGBMClassifier()
    model.booster._load_model(tmp.name)
  return model


def load_pytorch_model(file_obj):
  import torch
  import tempfile
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
    tmp.write(file_obj.read())
    tmp.flush()
    model = torch.load(tmp.name)
  return model

def load_keras_model(file_obj):
  from tensorflow import keras
  import tempfile
  with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
    tmp.write(file_obj.read())
    tmp.flush()
    model = keras.models.load_model(tmp.name)
  return model

def load_model(file_obj, filename):
  ext = os.path.splitext(filename)[-1].lower()
  if ext in [".pkl","joblib"]:
    return load_sklearn_model(file_obj)
  elif ext == ".json":
    return load_xgboost_model(file_obj)
  elif ext == ".txt":
    return load_lightgbm_model(file_obj)
  elif ext == ".pt":
    return load_pytorch_model(file_obj)
  elif ext == ".h5":
    return load_keras_model(file_obj)
  else:
    raise ValueError(f"Unsupported file extension: {ext}")
