import io
import os
import time
from typing import List

import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torchvision import transforms

from src.types.response import CarTypeClassProbabilities, CarTypePredictionResponse


class CarTypeModelService:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.labels = ["Convertible", "Coupe", "Hatchback", "Pick-Up", "SUV", "Sedan", "VAN"]
        self.class_to_idx = {self.labels[i]: i for i in range(len(self.labels))}
        self.idx_to_class = {i: self.labels[i] for i in range(len(self.labels))}

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        print(f"กำลังโหลดโมเดลจาก {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = self._build_model().to(self.device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError("ไฟล์โมเดลไม่มีค่าน้ำหนักโมเดล (model_state_dict)")

        self.model.eval()
        print("โหลดโมเดลสำเร็จ")

    def _build_model(self) -> nn.Module:
        model = models.efficientnet_b3(weights=None)

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(self.labels)),
        )

        return model

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    async def predict(self, image_bytes: bytes) -> CarTypePredictionResponse:
        start_time = time.time()
        if self.model is None:
            raise RuntimeError("โมเดลยังไม่ถูกโหลด กรุณาเรียก load_model ก่อน")

        image_tensor = self.preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]

        pred_idx = torch.argmax(probabilities).item()
        pred_class = self.idx_to_class[pred_idx]
        confidence = probabilities[pred_idx].item()

        class_probs: List[CarTypeClassProbabilities] = [
            {
                "name": self.idx_to_class[i],
                "prob": prob.item(),
            }
            for i, prob in enumerate(probabilities)
        ]

        processing_time = time.time() - start_time

        return CarTypePredictionResponse(
            class_name=pred_class,
            confidence=confidence,
            class_probabilities=class_probs,
            processing_time=processing_time,
        )


MODEL_PATH = "src/models/car_classification_best_model.pth"
cartype_model_service = CarTypeModelService(MODEL_PATH)
