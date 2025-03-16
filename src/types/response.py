from enum import Enum
from typing import List, Literal, TypedDict

from pydantic import BaseModel, Field


class CarTypeClassProbabilities(TypedDict):
    name: str
    prob: float


class CarTypePredictionResponse(BaseModel):
    class_name: str
    confidence: float
    class_probabilities: List[CarTypeClassProbabilities]
    processing_time: float


Gender_ = Literal["Male", "Female"]
FamilyObesityHistory = Literal["Yes", "No"]
JunkFoodIntake = Literal["High", "Medium", "Low"]
MedicalCondition = Literal["Yes", "No"]


class ObesityPredictionInput(BaseModel):
    Age: int = Field(..., description="อายุในปี", example=35, ge=0, le=120)
    Gender: Gender_ = Field(..., description="เพศ", example="Male")
    Height: float = Field(..., description="ส่วนสูงในเมตร", example=1.75, gt=0, lt=3)
    Weight: float = Field(..., description="น้ำหนักในกิโลกรัม", example=85.0, gt=0, lt=500)
    BMI: float = Field(..., description="ดัชนีมวลกาย", example=27.8, gt=0, lt=100)
    Daily_Calories: int = Field(..., description="แคลอรี่ที่บริโภคต่อวัน", example=2500, ge=0, le=10000)
    Exercise_Frequency: int = Field(..., description="ความถี่ในการออกกำลังกายต่อสัปดาห์", example=2, ge=0, le=7)
    Family_Obesity_History: FamilyObesityHistory = Field(..., description="ประวัติความอ้วนในครอบครัว", example="Yes")
    Junk_Food_Intake: JunkFoodIntake = Field(..., description="ความถี่ในการบริโภคอาหารขยะ", example="Occasional")
    Medical_Condition: MedicalCondition = Field(..., description="มีโรคประจำตัวหรือไม่", example="No")

    class Config:
        schema_extra = {
            "example": {
                "Age": 35,
                "Gender": "Male",
                "Height": 1.75,
                "Weight": 85.0,
                "BMI": 27.8,
                "Daily_Calories": 2500,
                "Exercise_Frequency": 2,
                "Family_Obesity_History": "Yes",
                "Junk_Food_Intake": "High",
                "Medical_Condition": "No",
            }
        }


class ObesityPredictionResponse(BaseModel):
    obesity_level: str
    probability: float
    features_importance: dict
    processing_time: float
