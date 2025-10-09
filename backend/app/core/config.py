from pydantic_settings import BaseSettings
from pydantic import PostgresDsn, field_validator
from typing import Optional

class Settings(BaseSettings):
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str
    DB_NAME: str
    DATABASE_URL: Optional[str] = None
    API_KEY: str
    UPLOAD_DIRECTORY: str = "uploads"  
    
    @field_validator("DATABASE_URL", mode="before")
    def assemble_db_url(cls, v, values):
        if v:
            return v
        return PostgresDsn.build(
            scheme="postgresql",
            username=values.data["DB_USER"],
            password=values.data["DB_PASSWORD"],
            host=values.data["DB_HOST"],
            port=values.data["DB_PORT"],
            path=f"/{values.data['DB_NAME']}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()