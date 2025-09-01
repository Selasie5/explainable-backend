import numpy as np

def get_top_contributors(X_row, shap_row, feature_names, pred_class, n=3):
    abs_shap = np.abs(shap_row)
    top_idx = np.argsort(abs_shap)[::-1][:n]
    total = np.sum(abs_shap)
    contributors = []
    for idx in top_idx:
        val = X_row[idx]
        shap_val = shap_row[idx]
        direction = (
            f"pushes_toward_{pred_class}" if shap_val > 0 else f"pushes_away_from_{pred_class}"
        )
        percent = 100 * abs(shap_val) / total if total else 0
        contributors.append({
            "feature": feature_names[idx],
            "value": val,
            "shap_value": float(shap_val),
            "direction": direction,
            "percent_of_local_effect": percent
        })
    return contributors

def summarize_local(pred, base, top):
    movers = []
    for t in top[:3]:
        arrow = "↑" if t["shap_value"] > 0 else "↓"
        movers.append(f"{t['feature']}={t['value']} ({arrow}{abs(t['shap_value']):.2f})")
    return (
        f"This prediction is {pred:.2f}, up from a base rate of {base:.2f}. "
        f"Key drivers: {', '.join(movers)}."
    )

def summarize_global(global_imp):
    sorted_feats = sorted(global_imp.items(), key=lambda x: -x[1])
    top_feats = [f"{k} ({v:.2f})" for k, v in sorted_feats[:3]]
    return (
        f"Across the analyzed rows, {', '.join(top_feats)} most strongly influence predictions."
    )
