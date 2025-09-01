# Explainable AI Backend Engine

This project provides an API and CLI for generating explainable model visualizations and narratives, including SHAP, LIME, and Integrated Gradients, to interpret and explain machine learning model predictions. It is containerized using Docker for easy deployment and reproducibility.

## Features

- Accepts structured input data and returns SHAP, LIME, or Integrated Gradients explanations
- Supports tabular, text, and image models (with extensibility for others)
- Dockerized for scalable and environment-agnostic deployment
- Saves plots to disk and serves results via API and CLI
- Extensible: plug in your own models and explanation methods
- Batch and single-row explanation support
- Human-friendly, narrative-rich JSON and HTML report outputs

---

## Tech Stack

- Python 3.10+
- FastAPI
- SHAP, LIME, Captum (Integrated Gradients)
- Scikit-learn / XGBoost / LightGBM / PyTorch / TensorFlow (optional)
- Matplotlib / Plotly
- Docker & Docker Compose

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/shap-explainer-backend.git
cd shap-explainer-backend
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the API server

```bash
uvicorn main:app --reload
```

### 4. Run CLI explanations

```bash
python cli_explain.py --model <model_file> --data <data_file> --method shap|lime|integrated_gradients
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
