"""
Model Evaluation API Routes

This module provides endpoints for evaluating the impact of generated features
by comparing baseline vs upgraded model performance.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from uuid import UUID
import pandas as pd

from app.core.dependencies import get_db
from app.models import database_models
from app.services.model_evaluation import ModelEvaluationService

router = APIRouter()


class EvaluationConfig(BaseModel):
    """Configuration for running a model evaluation experiment."""
    dataset_id: UUID
    run_id: UUID


@router.post("/evaluation/run")
def run_evaluation(
    config: EvaluationConfig,
    db: Session = Depends(get_db)
):
    """
    Run a model evaluation experiment comparing baseline vs upgraded models.
    
    This endpoint:
    1. Loads the dataset and feature generation run
    2. Creates baseline model (without generated features)
    3. Creates upgraded model (with generated features)
    4. Compares performance and returns results
    
    Args:
        config: Configuration containing dataset_id and run_id
        db: Database session
        
    Returns:
        Dictionary containing:
        - baseline_score: Performance without generated features
        - upgraded_score: Performance with generated features
        - improvement: Difference between upgraded and baseline
        - metric: The metric used (accuracy or rmse)
        - num_original_features: Count of original features
        - num_generated_features: Count of generated features
        - num_total_features: Total features in upgraded model
    """
    # Step 1: Load the dataset from the database
    dataset = db.query(database_models.Dataset).filter(
        database_models.Dataset.id == config.dataset_id
    ).first()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Step 2: Load the raw dataframe from disk
    try:
        dataframe = pd.read_csv(dataset.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read dataset file: {e}")
    
    # Step 3: Load the feature generation run from the database
    run = db.query(database_models.FeatureGenerationRun).filter(
        database_models.FeatureGenerationRun.id == config.run_id
    ).first()
    
    if not run:
        raise HTTPException(status_code=404, detail="Feature generation run not found")
    
    # Step 4: Extract problem_type and target_column from run parameters
    problem_type = run.parameters.get("problem_type")
    target_column = run.parameters.get("target_column")
    
    if not problem_type or not target_column:
        raise HTTPException(
            status_code=400, 
            detail="Run parameters missing problem_type or target_column"
        )
    
    # Step 5: Create the ModelEvaluationService
    evaluation_service = ModelEvaluationService(
        problem_type=problem_type,
        target_column=target_column
    )
    
    # Step 6: Run the comparison experiment
    try:
        results = evaluation_service.run_comparison(
            dataframe=dataframe,
            dataset_id=config.dataset_id,
            run_id=config.run_id,
            db=db
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
