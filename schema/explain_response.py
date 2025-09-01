from typing import List, Dict, Optional, Any
from pydantic import BaseModel

class Contributor(BaseModel):
    feature: str
    value: Any
    shap_value: float
    direction: str
    percent_of_local_effect: float

class PredictionContext(BaseModel):
    row_id: str
    predicted_class: str
    class_probabilities: Dict[str, float]
    decision_threshold: float
    base_value: float
    output_scale: str

class LocalExplanation(BaseModel):
    top_contributors: List[Contributor]
    sum_shap: float

class GlobalExplanation(BaseModel):
    feature_importance_mean_abs: Dict[str, float]
    n_samples: int
    notes: Optional[str] = None

class ArtifactURLs(BaseModel):
    summary_plot_url: str
    force_plot_html_url: str
    dependence_plot_urls: Optional[Dict[str, str]] = None

class NaturalLanguage(BaseModel):
    global_summary: str
    local_summary: str

class WhatIf(BaseModel):
    feature: str
    suggested_change: str
    estimated_effect: str
    method: str
    disclaimer: str

class Provenance(BaseModel):
    data_source: str
    model_source: str
    timestamp_utc: str
    explain_algo: str

class ModelInfo(BaseModel):
    type: str
    version: str
    classes: List[str]

class ExplainResponse(BaseModel):
    status: str
    message: str
    model: ModelInfo
    prediction: PredictionContext
    local_explanation: LocalExplanation
    global_explanation: GlobalExplanation
    artifacts: ArtifactURLs
    natural_language: NaturalLanguage
    what_if: List[WhatIf]
    provenance: Provenance
