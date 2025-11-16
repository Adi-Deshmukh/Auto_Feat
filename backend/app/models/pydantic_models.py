from pydantic import BaseModel, ConfigDict
from typing import Optional, Literal, List
from uuid import UUID
from datetime import datetime
from typing import List

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
    problem_type: Literal["regression", "classification"]
    drop_columns: Optional[List[str]] = None

    model_config = ConfigDict(from_attributes=True)


class GeneratedFeatureResponse(BaseModel):
    id: UUID
    run_id: UUID
    expression: str
    fitness: Optional[float] = None
    feature_names: Optional[List[str]] = None

    model_config = ConfigDict(from_attributes=True)


class RunResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    run_time: datetime
    parameters: dict
    generated_features: List[GeneratedFeatureResponse] = []

    model_config = ConfigDict(from_attributes=True)
