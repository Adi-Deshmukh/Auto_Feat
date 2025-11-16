import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, func, Float
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
    feature_generation_runs = relationship("FeatureGenerationRun", back_populates="dataset")


class Project(Base):
    __tablename__ = "projects" 

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    description = Column(String)
    domain = Column(String(100))
    status = Column(String(50), default="active")
    created_at = Column(DateTime, server_default=func.now())
    datasets = relationship("Dataset", back_populates="project")


class FeatureGenerationRun(Base):
    __tablename__ = "feature_generation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False)
    run_time = Column(DateTime, server_default=func.now())
    parameters = Column(JSONB)  # Store GP parameters like population_size, generations, etc.
    
    # Relationships
    dataset = relationship("Dataset", back_populates="feature_generation_runs")
    generated_features = relationship("GeneratedFeature", back_populates="run")


class GeneratedFeature(Base):
    __tablename__ = "generated_features"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("feature_generation_runs.id"), nullable=False)
    expression = Column(String, nullable=False)  # The mathematical formula
    fitness = Column(Float)  # The fitness score
    feature_names = Column(JSONB)  # List of feature names used in the expression
    
    # Relationship
    run = relationship("FeatureGenerationRun", back_populates="generated_features")

    