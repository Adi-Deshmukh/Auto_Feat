from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.dependencies import get_db
from app.models import database_models
import pandas as pd
from data_visualizer import AnalysisReport, Settings

router = APIRouter()

@router.post("/profiling/{dataset_id}", response_model=dict)
def generate_profiling_report(dataset_id: str, db: Session = Depends(get_db)):
    # Find the dataset by ID
    dataset = db.query(database_models.Dataset).filter(database_models.Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Read the dataset file into a DataFrame
    try:
        df = pd.read_csv(dataset.filepath)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading dataset: {str(e)}")
    
    # Configure report settings for faster, minimal analysis
    report_settings = Settings(
        minimal=False,              # Set to True for faster, minimal analysis
        top_n_values=5,            # Show top 5 values in categorical columns
        skewness_threshold=2.0,    # Tolerance for skewness alerts
        outlier_method='iqr',      # Outlier detection method: 'iqr' or 'zscore'
        outlier_threshold=1.5,     # IQR multiplier for outlier detection
        duplicate_threshold=5.0,   # Percentage threshold for duplicate alerts
        text_analysis=False,       # Enable word frequency analysis for text columns
        include_plots = False,
        include_correlations_plots= False,    # Disable correlation plots, only return correlation JSON
        include_correlations_json = True,
        include_sample_data = False
    )
    
    # Generate the analysis report with settings
    
    try:
        report = AnalysisReport(df, settings=report_settings)
        results = report.analyse()
        dataset.profiling_report = results # Store the report in the database
        db.commit()
        return {"message": "Profiling report generated and saved successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
    
    # Get the profiling report using project ID
@router.get("/profiling/{dataset_id}", response_model=dict)
def get_profiling_report(dataset_id: str, db: Session = Depends(get_db)):
    # Find the dataset by ID
    dataset = db.query(database_models.Dataset).filter(database_models.Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Return the profiling report
    return {"profiling_report": dataset.profiling_report}


