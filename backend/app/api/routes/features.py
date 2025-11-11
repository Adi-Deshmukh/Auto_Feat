from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

from app.core.dependencies import get_db
from app.models import database_models, pydantic_models
from app.services.genetic_programming import GeneticProgrammingService

router = APIRouter()

@router.post("/projects/{project_id}/datasets/{dataset_id}/generate-features", response_model=list)
def generate_features(
    project_id: str,
    dataset_id: str,
    config: pydantic_models.FeatureGenerationConfig, 
    db: Session = Depends(get_db)
):
    # Step 1: Find the dataset record in the database.
    dataset = db.query(database_models.Dataset).filter(database_models.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Step 2: Load the actual data file from disk.
    try:
        dataframe = pd.read_csv(dataset.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset file: {e}")

    # Step 3: Create an instance of our service and run it.
    gp_service = GeneticProgrammingService(target_column=config.target_column)
    best_features = gp_service.run(dataframe)

    # Step 4: Return the list of best features found.
    return best_features