from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine, Base
from app.api.routes import projects, upload, profiling
from app.core.dependencies import get_db    

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AutoFeat",
    description="An intelligent feature engineering platform using genetic programming.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/db-test")
def test_db_connection(db: Session = Depends(get_db)):
    return {"message": "Database connection successful"}


# Api endpoints
app.include_router(projects.router, tags=["projects"])
app.include_router(upload.router, tags=["datasets"])
app.include_router(profiling.router, tags=["profiling"])

