from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from datetime import datetime

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    domain: Optional[str] = None

class Project(ProjectCreate):
    id: UUID
    created_at: datetime

    class Config:
        orm_mode = True