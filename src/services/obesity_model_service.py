import os
import pickle
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.types.response import ObesityPredictionResponse


class ObesityModelService:
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.models: Optional[Dict[str, Any]] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Optional[Dict[str, LabelEncoder]] = None
        self.target_encoder: Optional[LabelEncoder] = None
        self.svm_model: Optional[SVC] = None
        self.knn_model: Optional[KNeighborsClassifier] = None
        self.voting_clf: Optional[VotingClassifier] = None

        self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        print(f"กำลังโหลดโมเดลจาก {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ไม่พบไฟล์โมเดลที่ {model_path}")

        with open(model_path, "rb") as file:
            self.models = pickle.load(file)

        self.scaler = self.models.get("scaler")
        self.label_encoders = self.models.get("label_encoders")
        self.target_encoder = self.models.get("target_encoder")
        self.svm_model = self.models.get("svm")
        self.knn_model = self.models.get("knn")
        self.voting_clf = self.models.get("voting")

        if not all([self.scaler, self.label_encoders, self.target_encoder, self.voting_clf]):
            raise ValueError("โมเดลไม่สมบูรณ์ กรุณาตรวจสอบไฟล์โมเดล")

        print("โหลดโมเดลสำเร็จ")

    def _preprocess_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        for key, value in input_data.items():
            if hasattr(value, "value"):
                input_data[key] = value.value

        df = pd.DataFrame([input_data])

        categorical_columns = ["Gender", "Family_Obesity_History", "Junk_Food_Intake", "Medical_Condition"]
        for col in categorical_columns:
            if col in df.columns:
                unique_values = self.label_encoders[col].classes_
                for val in df[col].unique():
                    if val not in unique_values:
                        raise ValueError(f"ค่า '{val}' ใน {col} ไม่มีอยู่ในโมเดล (มีเฉพาะ {list(unique_values)})")

                df[col] = self.label_encoders[col].transform(df[col])

        scaled_data = self.scaler.transform(df)
        return scaled_data

    def _get_feature_importance(self) -> dict:
        if hasattr(self.svm_model, "coef_"):
            importances = np.abs(self.svm_model.coef_).mean(axis=0)
            feature_names = [
                "Age",
                "Gender",
                "Height",
                "Weight",
                "BMI",
                "Daily_Calories",
                "Exercise_Frequency",
                "Family_Obesity_History",
                "Junk_Food_Intake",
                "Medical_Condition",
            ]
            importances = importances / importances.sum() * 100
            return {feature_names[i]: float(importances[i]) for i in range(len(feature_names))}
        else:
            return {}

    async def predict(self, input_data: Dict[str, Any]) -> ObesityPredictionResponse:
        start_time = time.time()

        if self.models is None:
            raise RuntimeError("โมเดลยังไม่ถูกโหลด กรุณาเรียก load_model ก่อน")

        processed_data = self._preprocess_data(input_data)
        prediction_encoded = self.voting_clf.predict(processed_data)
        pred_idx = prediction_encoded[0]
        pred_class = self.target_encoder.inverse_transform([pred_idx])[0]

        try:
            svm_proba = self.svm_model.predict_proba(processed_data)[0]
            confidence = svm_proba[pred_idx]
        except:
            try:

                knn_proba = self.knn_model.predict_proba(processed_data)[0]
                confidence = knn_proba[pred_idx]
            except:
                confidence = 1.0

        feature_importance = self._get_feature_importance()
        processing_time = time.time() - start_time

        return ObesityPredictionResponse(
            obesity_level=pred_class,
            probability=float(confidence),
            features_importance=feature_importance,
            processing_time=processing_time,
        )


MODEL_PATH = "src/models/obesity_prediction_models.pkl"
obesity_model_service = ObesityModelService(MODEL_PATH)
