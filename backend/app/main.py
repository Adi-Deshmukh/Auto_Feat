from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine, Base
import app.models.database_models  # Import models to create tables

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/db-test")
def test_db_connection(db: Session = Depends(get_db)):
    return {"message": "Database connection successful"}
