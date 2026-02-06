import base64
import functools
import io
import os
import pickle
import zipfile
from pathlib import Path

import timm
import torch
from PIL import Image
from ultralytics import YOLO

from config import settings
from s3_utils import s3


def base64_to_image(base64_string: str):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image


class LoadModel:
    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_detector_model():
        path = settings.APP_YOLO_WEIGHT
        if not os.path.exists(path):
            local_dir = Path(path)
            local_dir = local_dir.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            s3.download_files_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                file_path=os.path.join(
                    settings.AWS_S3_FILEPATH_PREFIX, settings.AWS_S3_MODEL_PATH
                ),
                local_dir=path,
            )
        model = YOLO(path)
        return model.to(settings.DEVICE)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_table_detector_model():
        path = "./model/detector/table/model_table.pt"
        if not os.path.exists(path):
            local_dir = Path(path)
            local_dir = local_dir.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            s3.download_files_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                file_path=os.path.join(
                    settings.AWS_S3_FILEPATH_PREFIX,
                    "model/detector/table/model_table.pt",
                ),
                local_dir=path,
            )
        model_table = YOLO(path)
        model_table.overrides["conf"] = 0.25
        model_table.overrides["iou"] = 0.45
        model_table.overrides["agnostic_nms"] = False
        model_table.overrides["max_det"] = 1000
        return model_table.to(settings.DEVICE)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_model_extraction(model_name) -> torch.nn.Sequential:
        base_model = timm.create_model(model_name, pretrained=True)
        model = torch.nn.Sequential(*list(base_model.children())[:-1])
        return model.to(settings.DEVICE)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_shap_extractor_model():
        path = "./model/detector/shape/best_shap_detection.pt"
        if not os.path.exists(path):
            local_dir = Path(path)
            local_dir = local_dir.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            s3.download_files_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                file_path=os.path.join(
                    settings.AWS_S3_FILEPATH_PREFIX,
                    "model/detector/shape/best_shap_detection.pt",
                ),
                local_dir=path,
            )

        return YOLO(path).to(settings.DEVICE)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_classification_model():
        path = "./model/classification/best_classification.pt"
        if not os.path.exists(path):
            local_dir = Path(path)
            local_dir = local_dir.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            s3.download_files_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                file_path=os.path.join(
                    settings.AWS_S3_FILEPATH_PREFIX,
                    "model/classification/best_classification.pt",
                ),
                local_dir=path,
            )
        return YOLO(path).to(settings.DEVICE)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_paddleocr():
        path = "PaddleOCR.zip"
        if not os.path.exists(path):
            s3.download_files_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                file_path=os.path.join(settings.AWS_S3_FILEPATH_PREFIX, "OCR", path),
                local_dir=path,
            )
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(""))

        path = "/model/ocr/ser/inference"
        if not os.path.exists(path):
            local_dir = Path(path)
            local_dir = local_dir.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            s3.download_folder_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                folder_path=os.path.join(
                    settings.AWS_S3_FILEPATH_PREFIX, "model/ocr/ser_v1/inference"
                ),
                local_dir=path,
            )

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def load_word_classification_model():
        path = "./model/word_classification/text_classification_pipeline.pkl"
        if not os.path.exists(path):
            local_dir = Path(path)
            local_dir = local_dir.parent
            local_dir.mkdir(parents=True, exist_ok=True)
            s3.download_files_to_local(
                bucket_name=settings.AWS_S3_BUCKET,
                file_path=os.path.join(
                    settings.AWS_S3_FILEPATH_PREFIX,
                    "model/word_classification/text_classification_pipeline.pkl",
                ),
                local_dir=path,
            )

        with open(path, "rb") as f:
            loaded_pipeline = pickle.load(f)

        return loaded_pipeline


model_loader = LoadModel()
model_loader.load_word_classification_model()
