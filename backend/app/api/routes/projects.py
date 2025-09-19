from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from fastapi import HTTPException
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

@router.get("/projects/", response_model=List[pydantic_models.Project])
def read_projects(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    projects = db.query(database_models.Project).offset(skip).limit(limit).all()
    return projects 

@router.get("/projects/{project_id}", response_model=pydantic_models.Project)
def read_project(project_id: str, db: Session = Depends(get_db)):
    project = db.query(database_models.Project).filter(database_models.Project.id == project_id).first()
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project
