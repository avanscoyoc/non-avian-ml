from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import uvicorn
from main import main
from config import get_default_args

app = FastAPI(
    title="Non-Avian ML Model API", description="API for running ML model evaluations"
)


class EvaluationRequest(BaseModel):
    model_name: str
    species_list: List[str]
    training_size: float = 0.8
    batch_size: int = 32
    n_folds: int = 5
    random_seed: int = 42
    datatype: str = "features"
    datapath: Optional[str] = None
    results_path: Optional[str] = None
    gcs_bucket: Optional[str] = None


class EvaluationResponse(BaseModel):
    results: Dict[str, float]
    fold_scores: Dict[str, List[float]]
    config: Dict[str, Any]


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        # Run the evaluation
        results, fold_scores = main(**request)
        # Return structured response
        return {
            "results": results,
            "fold_scores": fold_scores,
            "config": {
                "model_name": request.model_name,
                "datatype": request.datatype,
                "training_size": request.training_size,
                "species_list": request.species_list,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Non-Avian ML Model API is running. Use /evaluate endpoint to evaluate models."
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
