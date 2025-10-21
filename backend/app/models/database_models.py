import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.core.database import Base
import pandas as pd


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String, nullable=False)
    content_type = Column(String(100))
    row_count = Column(Integer)
    upload_time = Column(DateTime, server_default=func.now())
    profiling_report = Column(JSONB)  # Store profiling report as JSON
    project = relationship("Project", back_populates="datasets")


class Project(Base):
    __tablename__ = "projects" 

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(String)
    domain = Column(String(100))
    status = Column(String(50), default="active")
    created_at = Column(DateTime, server_default=func.now())
    datasets = relationship("Dataset", back_populates="project")
    