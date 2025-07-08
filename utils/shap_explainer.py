import shap
import matplotlib.pyplot as plt
import os
import uuid
import numpy as np
import logging
from typing import Union, Dict, List
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from starlette.responses import StreamingResponse
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create explanations output directory
EXPLANATION_OUTPUT_DIR = "explanations"
os.makedirs(EXPLANATION_OUTPUT_DIR, exist_ok=True)

def is_tree_based_model(model: BaseEstimator) -> bool:
    """Check if the model is tree-based"""
    return isinstance(model, (XGBClassifier, LGBMClassifier,
                            RandomForestClassifier, GradientBoostingClassifier))

def generate_shap_explanations(model: BaseEstimator, data: Union[np.ndarray, 'pd.DataFrame']) -> Dict:
    """
    Generate SHAP explanations for a given model and dataset
    
    Args:
        model: Trained machine learning model
        data: Input data (pandas DataFrame or numpy array)
    
    Returns:
        Dictionary containing:
        - summary_plot_path: Base64 encoded image (temporary field name)
        - force_plot_path: Path to force plot HTML
        - feature_importance: Dictionary of feature importances
        - status: Success/error status
        - message: Detailed status message
    """
    result = {
        "summary_plot_path": None,  # Will contain base64 image
        "force_plot_path": None,
        "feature_importance": {},
        "status": "error",
        "message": ""
    }
    
    try:
       # Convert data to numpy array if needed and get feature names
        X = data.values if hasattr(data, 'values') else np.array(data)
        feature_names = (data.columns.tolist() if hasattr(data, 'columns') 
                        else [f'feature_{i}' for i in range(X.shape[1])])

        # Initialize appropriate SHAP explainer
        if is_tree_based_model(model):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (LogisticRegression, RidgeClassifier)):
            # Handle feature names warning for sklearn models
            if hasattr(data, 'columns') and not hasattr(model, 'feature_names_in_'):
                logger.warning("Model fitted without feature names. Using provided feature names.")
                # Create a new model with feature names if possible (sklearn >= 1.0)
                if hasattr(model, 'set_output'):
                    model.set_output(transform="pandas")
            masker = shap.maskers.Independent(X, max_samples=100)
            explainer = shap.LinearExplainer(model, masker)
        else:
            explainer = shap.Explainer(model, X)

        # Compute SHAP values
        shap_values = explainer(X)

        # Convert to Explanation object if needed
        if not isinstance(shap_values, shap.Explanation):
            base_value = explainer.expected_value
            if isinstance(base_value, np.ndarray) and len(base_value) == 1:
                base_value = base_value[0]
                
            shap_values = shap.Explanation(
                values=shap_values.values if hasattr(shap_values, 'values') else shap_values,
                base_values=base_value,
                data=X,
                feature_names=feature_names
            )

        # Generate summary plot (base64)
        buf = io.BytesIO()
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        result["summary_plot_path"] = base64.b64encode(buf.read()).decode('utf-8')

        # Generate force plot
        result["force_plot_path"] = generate_force_plot(shap_values, explainer, feature_names, model)

        # Compute feature importances
        feature_importances = compute_feature_importances(shap_values)
        result["feature_importance"] = {
            feature: float(importance)
            for feature, importance in zip(feature_names, feature_importances)
        }

        result["status"] = "success"
        return result

    except Exception as e:
        logger.error(f"SHAP explanation generation failed: {str(e)}")
        result["message"] = f"Error: {str(e)}"
        return result

def generate_force_plot(shap_values: shap.Explanation, 
                       explainer: shap.Explainer, 
                       feature_names: List[str], 
                       model: BaseEstimator) -> Union[str, None]:
    """
    Generate force plot with comprehensive error handling and fallbacks
    
    Args:
        shap_values: SHAP values Explanation object
        explainer: SHAP explainer instance
        feature_names: List of feature names
        model: The trained model
    
    Returns:
        Path to saved force plot HTML or None if generation failed
    """
    force_path = None
    expected_value = explainer.expected_value
    
    # Convert expected_value to proper format
    if isinstance(expected_value, np.ndarray) and len(expected_value) == 1:
        expected_value = expected_value[0]
    
    try:
        # Handle different model types and SHAP value shapes
        if len(shap_values.shape) == 3:  # Multiclass case
            predicted_class = np.argmax(shap_values.values[0].sum(axis=0))
            ev = expected_value[predicted_class] if isinstance(expected_value, np.ndarray) else expected_value
            values = shap_values.values[0, predicted_class]
        else:  # Binary or regression
            ev = expected_value[0] if isinstance(expected_value, np.ndarray) else expected_value
            values = shap_values.values[0]
        
        # Generate the force plot with proper parameter ordering
        force_plot = shap.plots.force(
            base_value=ev,
            shap_values=values,
            features=shap_values.data[0],
            feature_names=feature_names,
            matplotlib=False
        )
        
        # Save the plot
        force_path = os.path.join(EXPLANATION_OUTPUT_DIR, f"{uuid.uuid4()}_force_plot.html")
        shap.save_html(force_path, force_plot)
        
    except Exception as e:
        logger.error(f"Primary force plot generation failed: {str(e)}")
        try:
            # Fallback 1: Try without feature names
            force_plot = shap.plots.force(
                base_value=ev,
                shap_values=values,
                matplotlib=False
            )
            force_path = os.path.join(EXPLANATION_OUTPUT_DIR, f"{uuid.uuid4()}_force_plot.html")
            shap.save_html(force_path, force_plot)
        except Exception as e:
            logger.error(f"Fallback 1 force plot generation failed: {str(e)}")
            try:
                # Fallback 2: Try simplest possible force plot
                force_plot = shap.plots.force(
                    base_value=expected_value[0] if isinstance(expected_value, np.ndarray) else expected_value,
                    shap_values=shap_values.values[0],
                    matplotlib=False
                )
                force_path = os.path.join(EXPLANATION_OUTPUT_DIR, f"{uuid.uuid4()}_force_plot.html")
                shap.save_html(force_path, force_plot)
            except Exception as e:
                logger.error(f"Fallback 2 force plot generation failed: {str(e)}")
    
    return force_path

def compute_feature_importances(shap_values: shap.Explanation) -> np.ndarray:
    """Compute feature importances from SHAP values"""
    if len(shap_values.shape) == 3:  # Multiclass
        return np.mean(np.abs(shap_values.values), axis=(0, 1))
    else:  # Binary or regression
        return np.mean(np.abs(shap_values.values), axis=0)
