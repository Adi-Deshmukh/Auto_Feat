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

class DatasetResponse(BaseModel):
    id: UUID
    project_id: UUID
    name: str
    filename: str
    content_type: str
    row_count: Optional[int] = None
    upload_time: datetime

    model_config = ConfigDict(from_attributes=True)


class FeatureGenerationConfig(BaseModel):
    target_column: str
    '''   
    enabled: bool = True
    methods: list[str] = ["add", "sub", "mul", "div", "sqrt", "log"]
    max_features: int = 10
    random_state: int = 0 '''

