import os

from dotenv import load_dotenv

load_dotenv()  # loads .env into environment variables


class Settings:
    def __init__(self):
        self.ELASTICSEARCH_ADDRESS = os.getenv("ELASTICSEARCH_ADDRESS")
        self.ELASTICSEARCH_USERNAME = os.getenv("ELASTICSEARCH_USERNAME")
        self.ELASTICSEARCH_PASSWORD = os.getenv("ELASTICSEARCH_PASSWORD")
        self.AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
        self.AWS_S3_UPLOADED_BUCKET = os.getenv("AWS_S3_UPLOADED_BUCKET")
        self.AWS_COGNITO_ID_POOL_ID = os.getenv("AWS_COGNITO_ID_POOL_ID")
        self.AWS_COGNITO_USER_POOL_ID = os.getenv("AWS_COGNITO_USER_POOL_ID")
        self.AWS_S3_FILEPATH_PREFIX = os.getenv("AWS_S3_FILEPATH_PREFIX")
        self.AWS_S3_MODEL_PATH = os.getenv("AWS_S3_MODEL_PATH")
        self.AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
        self.AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
        self.DEVICE = os.getenv("DEVICE", "cpu")
        self.PHYSICAL_OBJ_DATABASE_HOST = os.getenv("PHYSICAL_OBJ_DATABASE_HOST")
        self.PHYSICAL_OBJ_DATABASE_USER = os.getenv("PHYSICAL_OBJ_DATABASE_USER")
        self.PHYSICAL_OBJ_DATABASE_PASSWORD = os.getenv(
            "PHYSICAL_OBJ_DATABASE_PASSWORD"
        )
        self.PHYSICAL_OBJ_DATABASE = os.getenv("PHYSICAL_OBJ_DATABASE")
        self.APP_SEARCH_UPDATER_RESOURCE_PATH = os.getenv(
            "APP_SEARCH_UPDATER_RESOURCE_PATH"
        )


settings = Settings()
