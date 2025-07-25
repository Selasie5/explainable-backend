from fastapi import APIRouter, UploadFile, File, status, HTTPException
from fastapi.responses import JSONResponse
from utils.file_handler import process_uploaded_file
from utils.file_loader import validate_model_compatibility
from services.shap_explainer import generate_shap_explanations
from utils.logger import logger
AnalyzeRouter = APIRouter()

@AnalyzeRouter.post("/analyze")
async def analyze_files(csv: UploadFile = File(...), model: UploadFile = File(...)):
    try:
        logger.info(f"CSV file: {csv.filename}, Model file: {model.filename}")

       
        df, df_source = process_uploaded_file(csv)
        loaded_model, model_source = process_uploaded_file(model)

        # Validate model compatibility
        validate_model_compatibility(loaded_model, df)

        # Generate SHAP explanations
        shap_results = generate_shap_explanations(loaded_model, df)
        
        if shap_results["status"] != "success":
            raise HTTPException(status_code=400, detail=shap_results["message"])

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
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the files."
        )
