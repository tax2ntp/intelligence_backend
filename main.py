from src.config import config

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.app:app",
        host="127.0.0.1",
        port=config.PORT,
        reload=(config.ENVIRONMENT != "production"),
    )
