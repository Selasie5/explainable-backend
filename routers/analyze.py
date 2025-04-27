from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from utils.file_handler import process_uploaded_file
from utils.file_loader import validate_model_compatibility
from utils.shap_explainer import generate_shap_explanations

AnalyzeRouter = APIRouter()

@AnalyzeRouter.post("/analyze")
async def analyze_files(csv: UploadFile = File(...), model: UploadFile = File(...)):
    # Process uploaded files
    df, df_source = process_uploaded_file(csv)
    loaded_model, model_source = process_uploaded_file(model)

    # Validate model compatibility
    validate_model_compatibility(loaded_model, df)

    # Generate SHAP explanations
    shap_results = generate_shap_explanations(loaded_model, df)

    return JSONResponse(
        content={
            "message": "SHAP explanations generated successfully.",
            "csv_source": df_source,
            "model_source": model_source,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "model_type": type(loaded_model).__name__,
            "shap_summary_plot": shap_results["summary_plot_path"],
            "shap_force_plot": shap_results["force_plot_path"],
            "feature_importance": shap_results["feature_importance"]
        }
    )
