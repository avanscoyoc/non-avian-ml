from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import uvicorn
from src.main import main

app = FastAPI(
    title="Non-Avian ML Model API", description="API for running ML model evaluations"
)


class EvaluationRequest(BaseModel):
    model_name: str
    species_list: List[str]
    training_size: int = 10
    batch_size: Optional[int] = 32
    n_folds: Optional[int] = 5
    random_seed: Optional[int] = 42
    datatype: Optional[str] = "data"
    datapath: Optional[str] = "/workspaces/non-avian-ml-toy/results"
    results_path: Optional[str] = "/workspaces/non-avian-ml-toy/results"
    gcs_bucket: Optional[str] = "dse-staff/soundhub"


class EvaluationResponse(BaseModel):
    results: Dict[str, float]
    fold_scores: Dict[str, List[float]]
    config: Dict[str, Any]


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        # Run the evaluation
        results, fold_scores = main(**request.dict())
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
