import shap
import matplotlib.pyplot as plt
import os
import uuid
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

EXPLANATION_OUTPUT_DIR = "explanations"
os.makedirs(EXPLANATION_OUTPUT_DIR, exist_ok=True)

def is_tree_based_model(model):
    return isinstance(model,(XGBClassifier,LGBMClassifier,RandomForestClassifier, GradientBoostingClassifier))
    
def generate_shap_explanations(model, data):
   
    X = data if isinstance(data, np.ndarray) else data.values

    
    if is_tree_based_model(model):
        explainer = shap.TreeExplainer(model)
    else:
       from sklearn.linear_model import LogisticRegression, RidgeClassifier

    if isinstance(model, (LogisticRegression, RidgeClassifier)):
            explainer = shap.LinearExplainer(model, X, feature_dependence="independent")
    else:
            explainer = shap.Explainer(model, X)


    shap_values = explainer(X)

    # Save summary plot
    summary_path = os.path.join(EXPLANATION_OUTPUT_DIR, f"{uuid.uuid4()}_summary_plot.png")
    shap.summary_plot(shap_values, data, show=False)
    plt.savefig(summary_path, bbox_inches='tight')
    plt.clf()

    # Save force plot for first sample
    force_plot_html = shap.plots.force(explainer.expected_value, shap_values[0], matplotlib=False)
    force_path = os.path.join(EXPLANATION_OUTPUT_DIR, f"{uuid.uuid4()}_force_plot.html")
    shap.save_html(force_path, force_plot_html)

    # Feature importance (mean absolute SHAP values)
    feature_importances = np.abs(shap_values.values).mean(axis=0)
    importance_dict = {
        feature: float(importance)
        for feature, importance in zip(data.columns, feature_importances)
    }

    return {
        "summary_plot_path": summary_path,
        "force_plot_path": force_path,
        "feature_importance": importance_dict
    }
