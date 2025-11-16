from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

from app.core.dependencies import get_db
from app.models import database_models, pydantic_models
from app.services.genetic_programming import GeneticProgramming

router = APIRouter()

@router.post("/projects/{project_id}/datasets/{dataset_id}/generate-features", response_model=pydantic_models.RunResponse)
def generate_features(
    project_id: str,
    dataset_id: str,
    config: pydantic_models.FeatureGenerationConfig, 
    db: Session = Depends(get_db)
):
    # Strip whitespace from IDs to prevent UUID parsing errors
    dataset_id = dataset_id.strip()
    project_id = project_id.strip()
    
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
    gp_service = GeneticProgramming(
        target_column=config.target_column,
        problem_type=config.problem_type,
        drop_columns=config.drop_columns
    )
    run = gp_service.run(
        dataframe=dataframe,
        db=db,
        dataset_id=dataset.id
    )

    # Step 4: Return the FeatureGenerationRun object (the "receipt").
    return run


@router.get("/datasets/{dataset_id}/runs", response_model=list[pydantic_models.RunResponse])
def get_dataset_runs(
    dataset_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all feature generation runs for a specific dataset.
    
    This endpoint returns the history of all GP experiments run on a dataset,
    allowing users to see all their feature engineering attempts.
    """
    # Strip whitespace from ID
    dataset_id = dataset_id.strip()
    
    # Find the dataset
    dataset = db.query(database_models.Dataset).filter(database_models.Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Return all feature generation runs for this dataset
    return dataset.feature_generation_runs


@router.get("/runs/{run_id}/features", response_model=list[pydantic_models.GeneratedFeatureResponse])
def get_run_features(
    run_id: str,
    db: Session = Depends(get_db)
):
    """
    Get all generated features for a specific run.
    
    This endpoint returns the list of features discovered during a specific
    GP experiment, including their expressions and fitness scores.
    """
    # Strip whitespace from ID
    run_id = run_id.strip()
    
    # Find the run
    run = db.query(database_models.FeatureGenerationRun).filter(database_models.FeatureGenerationRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Feature generation run not found")
    
    # Return all generated features for this run
    return run.generated_features
