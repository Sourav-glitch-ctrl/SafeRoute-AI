from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str
    API_V1_PREFIX: str

    HOST: str
    PORT: int

    DEBUG: bool

    DATABASE_URL: str

    class Config:
        env_file = ".env"


settings = Settings()