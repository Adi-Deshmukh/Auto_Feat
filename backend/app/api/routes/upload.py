from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import os
import shutil
import pandas as pd
from app.core.database import SessionLocal
from app.core.dependencies import get_db
from app.core.config import settings
from app.models import database_models, pydantic_models
import uuid

router = APIRouter()
UPLOAD_DIRECTORY = settings.UPLOAD_DIRECTORY

@router.post("/projects/{project_id}/datasets/", response_model=pydantic_models.DatasetResponse)
def upload_dataset(project_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Verify the project exists
    project = db.query(database_models.Project).filter(database_models.Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Create upload directory if it doesn't exist
    if not os.path.exists(UPLOAD_DIRECTORY):
        os.makedirs(UPLOAD_DIRECTORY)

    # Generate unique file ID and save file to disk
    file_id = str(uuid.uuid4())
    file_location = os.path.join(UPLOAD_DIRECTORY, file_id + "_" + file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Calculate row count by reading the file
    row_count = None
    try:
        df = pd.read_csv(file_location)
        row_count = len(df)
    except Exception as e:
        # If we can't read the file, we'll just set row_count to None
        pass

    # Create Dataset record in database
    db_dataset = database_models.Dataset(
        id=file_id,
        project_id=project_id,
        name=file.filename,  # You can modify this to allow custom names
        filename=file.filename,
        filepath=file_location,
        content_type=file.content_type,
        row_count=row_count
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)

    return db_dataset


@router.get("/projects/{project_id}/datasets/", response_model=list[pydantic_models.DatasetResponse])
def get_project_datasets(project_id: str, db: Session = Depends(get_db)):
    """Get all datasets for a specific project."""
    project = db.query(database_models.Project).filter(database_models.Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    
    datasets = db.query(database_models.Dataset).filter(
        database_models.Dataset.project_id == project_id
    ).all()
    return datasets
