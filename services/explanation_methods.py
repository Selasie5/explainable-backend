import numpy as np

def lime_tabular_explanation(model, data, sample_idx=0, feature_names=None, class_names=None):
    from lime.lime_tabular import LimeTabularExplainer
    X = data.values if hasattr(data, 'values') else np.array(data)
    explainer = LimeTabularExplainer(
        X,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification"
    )
    exp = explainer.explain_instance(
        X[sample_idx],
        model.predict_proba
    )
    return exp.as_list(), exp.as_pyplot_figure()

def integrated_gradients_explanation(model, data, sample_idx=0):
    import torch
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model)
    input_tensor = torch.tensor(data[sample_idx:sample_idx+1], dtype=torch.float32)
    attr, _ = ig.attribute(input_tensor, target=0, return_convergence_delta=True)
    return attr.detach().numpy()
