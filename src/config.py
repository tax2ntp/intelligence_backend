from pydantic_settings import BaseSettings


class Config(BaseSettings):
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"
    FRONTEND_URL: str = ""
    FRONTEND_2_URL: str = ""
    PORT: int = 8080

    class Config:
        case_sensitive = True
        env_file = ".env"


config = Config()
