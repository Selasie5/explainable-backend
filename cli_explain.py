import argparse
import pandas as pd
import joblib
from services.explanation_methods import lime_tabular_explanation, integrated_gradients_explanation
from services.shap_explainer import generate_shap_explanations

def main():
    parser = argparse.ArgumentParser(description="Explain model predictions")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data", required=True, help="Path to CSV data")
    parser.add_argument("--method", choices=["shap", "lime", "integrated_gradients"], default="shap")
    parser.add_argument("--batch", action="store_true", help="Explain all rows")
    args = parser.parse_args()

    model = joblib.load(args.model)
    data = pd.read_csv(args.data)

    if args.method == "shap":
        result = generate_shap_explanations(model, data)
        print(result)
    elif args.method == "lime":
        for i in range(data.shape[0] if args.batch else 1):
            exp, fig = lime_tabular_explanation(model, data, i, feature_names=data.columns)
            print(f"Row {i} LIME explanation:", exp)
    elif args.method == "integrated_gradients":
        for i in range(data.shape[0] if args.batch else 1):
            ig = integrated_gradients_explanation(model, data.values, i)
            print(f"Row {i} Integrated Gradients:", ig)

if __name__ == "__main__":
    main()
