from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
import os
import shutil
from app.core.database import SessionLocal
from app.core.dependencies import get_db
from app.core.config import settings
from app.models import database_models, pydantic_models
import uuid

router = APIRouter()
UPLOAD_DIRECTORY = settings.UPLOAD_DIRECTORY

@router.post("/upload/", response_model=pydantic_models.FileUploadResponse) #file format for the output
def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not os.path.exists(UPLOAD_DIRECTORY): # creates a directory to save you files or saves in the existing directory
        os.makedirs(UPLOAD_DIRECTORY)

    file_id = str(uuid.uuid4()) # Generating a unique ID (UUID) for the file.
    file_location = os.path.join(UPLOAD_DIRECTORY, file_id + "_" + file.filename) #Creating a new, unique filename by prepending the ID to the original filename

    with open(file_location, "wb") as buffer: # Save the File
        shutil.copyfileobj(file.file, buffer)

    # Create a DB record and fileupload for storing the metadata

    db_file = database_models.FileUpload(
        id=file_id,
        filename=file.filename,
        filepath=file_location,
        content_type=file.content_type
    )
    db.add(db_file)
    db.commit()
    db.refresh(db_file)

    return pydantic_models.FileUploadResponse(
        id=db_file.id,
        filename=db_file.filename,
        content_type=db_file.content_type,
        upload_time=db_file.upload_time
    )
