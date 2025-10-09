from pydantic import BaseModel, ConfigDict
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

    model_config = ConfigDict(from_attributes=True)
        
class FileUploadResponse(BaseModel):
    id: UUID
    filename: str
    content_type: str
    upload_time: datetime

    model_config = ConfigDict(from_attributes=True)