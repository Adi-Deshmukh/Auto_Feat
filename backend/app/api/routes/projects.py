from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models import database_models, pydantic_models
from app.core.dependencies import get_db

router = APIRouter()



        
@router.post("/projects/", response_model=pydantic_models.Project)
def create_project(project: pydantic_models.ProjectCreate, db: Session = Depends(get_db)):
    db_project = database_models.Project(**project.model_dump())
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project        

