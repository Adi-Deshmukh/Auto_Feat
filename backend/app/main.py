from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from app.core.database import SessionLocal, engine, Base
from app.api.routes import projects, upload, profiling, features, evaluation
from app.core.dependencies import get_db    

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="AutoFeat",
    description="An intelligent feature engineering platform using genetic programming.",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:5173"
    ],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
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
app.include_router(features.router, tags=["features"])
app.include_router(evaluation.router, tags=["evaluation"])