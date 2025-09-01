from fastapi import APIRouter, UploadFile, File, status, HTTPException
from fastapi.responses import JSONResponse
from utils.file_handler import process_uploaded_file
from utils.file_loader import validate_model_compatibility
from services.shap_explainer import generate_shap_explanations
from services.explanation_methods import lime_tabular_explanation, integrated_gradients_explanation
from utils.logger import logger
AnalyzeRouter = APIRouter()

from fastapi import Query

@AnalyzeRouter.post("/analyze")
async def analyze_files(
    csv: UploadFile = File(...),
    model: UploadFile = File(...),
    method: str = Query("shap", enum=["shap", "lime", "integrated_gradients"]),
    batch: bool = Query(False)
):
    try:
        logger.info(f"CSV file: {csv.filename}, Model file: {model.filename}")
        df, df_source = process_uploaded_file(csv)
        loaded_model, model_source = process_uploaded_file(model)
        validate_model_compatibility(loaded_model, df)

        if method == "shap":
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
        elif method == "lime":
            results = []
            for i in range(df.shape[0] if batch else 1):
                exp, fig = lime_tabular_explanation(loaded_model, df, i, feature_names=df.columns)
                # Optionally, save or encode fig as needed
                results.append({"row": i, "lime_explanation": exp})
            return JSONResponse(content={"lime_results": results})
        elif method == "integrated_gradients":
            results = []
            for i in range(df.shape[0] if batch else 1):
                ig = integrated_gradients_explanation(loaded_model, df.values, i)
                results.append({"row": i, "integrated_gradients": ig.tolist()})
            return JSONResponse(content={"integrated_gradients_results": results})
    except HTTPException as e:
        logger.error(f"HTTP Exception: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing the files."
        )
