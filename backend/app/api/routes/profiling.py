from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.dependencies import get_db
from app.models import database_models
import pandas as pd
from data_visualizer import AnalysisReport, Settings, generate_html_report
import os
import tempfile
from datetime import datetime

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
    
    # Configure report settings for detailed analysis with plots
    report_settings = Settings(
        minimal=False,              # Full analysis with all features
        top_n_values=10,           # Show top 10 values in categorical columns
        skewness_threshold=2.0,    # Tolerance for skewness alerts
        outlier_method='iqr',      # Outlier detection method: 'iqr' or 'zscore'
        outlier_threshold=1.5,     # IQR multiplier for outlier detection
        duplicate_threshold=5.0,   # Percentage threshold for duplicate alerts
        text_analysis=True,        # Enable word frequency analysis for text columns
        include_plots=True,        # Enable distribution plots for numeric columns
        include_correlations_plots=True,    # Enable correlation heatmap
        include_correlations_json=True,     # Include correlation data as JSON
        include_sample_data=True   # Show sample data in report
    )
    
    # Generate the analysis report with settings
    
    try:
        report = AnalysisReport(df, settings=report_settings)
        
        # Get dictionary results
        results = report.analyse()
        
        # Generate HTML report to a temporary file then read it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tmp_file:
            temp_path = tmp_file.name
        
        try:
            generate_html_report(results, temp_path)
            
            # Read the HTML content
            with open(temp_path, 'r', encoding='utf-8') as f:
                html_report = f.read()
            
            # Clean up temp file
            os.unlink(temp_path)
            
            print(f"=== PROFILING GENERATION SUCCESS ===")
            print(f"Dataset ID: {dataset_id}")
            print(f"HTML length: {len(html_report)}")
            print(f"HTML preview (first 200 chars): {html_report[:200]}")
            
            # Store HTML in JSONB as a dictionary
            dataset.profiling_report = {
                "html": html_report,
                "generated_at": datetime.now().isoformat()
            }
            
            print(f"Before commit - profiling_report type: {type(dataset.profiling_report)}")
            print(f"Before commit - profiling_report has 'html' key: {'html' in dataset.profiling_report}")
            
            db.commit()
            db.refresh(dataset)
            
            print(f"After commit - profiling_report type: {type(dataset.profiling_report)}")
            print(f"After commit - profiling_report keys: {dataset.profiling_report.keys() if isinstance(dataset.profiling_report, dict) else 'Not a dict'}")
            print(f"After commit - has 'html' key: {'html' in dataset.profiling_report if isinstance(dataset.profiling_report, dict) else False}")
            print(f"===================================")
            
            return {
                "message": "Profiling report generated and saved successfully",
                "status": "success",
                "html_length": len(html_report)
            }
        finally:
            # Ensure temp file is deleted even if something goes wrong
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"=== PROFILING ERROR ===")
        print(f"Dataset ID: {dataset_id}")
        print(f"Error: {str(e)}")
        print(f"Traceback: {error_details}")
        print(f"=====================")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
    
    # Get the profiling report using project ID
@router.get("/profiling/{dataset_id}", response_model=dict)
def get_profiling_report(dataset_id: str, db: Session = Depends(get_db)):
    # Find the dataset by ID
    dataset = db.query(database_models.Dataset).filter(database_models.Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    print(f"=== PROFILING GET REQUEST ===")
    print(f"Dataset ID: {dataset_id}")
    print(f"Dataset found: {dataset is not None}")
    print(f"profiling_report exists: {dataset.profiling_report is not None}")
    
    if not dataset.profiling_report:
        print(f"ERROR: Profiling report is None or empty")
        raise HTTPException(status_code=404, detail="Profiling report not generated yet")

    # Extract HTML from JSONB structure
    print(f"Profiling report type: {type(dataset.profiling_report)}")
    print(f"Profiling report keys: {dataset.profiling_report.keys() if isinstance(dataset.profiling_report, dict) else 'Not a dict'}")
    
    html_content = dataset.profiling_report.get("html") if isinstance(dataset.profiling_report, dict) else dataset.profiling_report
    
    if not html_content:
        print(f"ERROR: HTML content is empty after extraction")
        print(f"Full profiling_report value: {str(dataset.profiling_report)[:500]}")
        raise HTTPException(status_code=404, detail="HTML report content is empty")
    
    print(f"HTML content length: {len(html_content) if html_content else 0}")
    print(f"HTML content type: {type(html_content)}")
    print(f"HTML preview (first 100 chars): {html_content[:100] if html_content else 'None'}")
    print(f"============================")
    
    # Return the profiling report with correct key
    return {"html_report": html_content}


