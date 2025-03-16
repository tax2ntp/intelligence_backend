import traceback

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.services import obesity_model_service
from src.services.cartype_model_service import cartype_model_service
from src.services.obesity_model_service import obesity_model_service
from src.types.response import (
    CarTypePredictionResponse,
    ObesityPredictionInput,
    ObesityPredictionResponse,
)

router = APIRouter()


@router.post("/cartype/predict", response_model=CarTypePredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="ไฟล์ต้องเป็นรูปภาพเท่านั้น")

    try:
        contents = await file.read()
        return await cartype_model_service.predict(contents)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")


@router.post("/obesity/predict", response_model=ObesityPredictionResponse)
async def predict_obesity(data: ObesityPredictionInput):
    try:
        input_data = data.model_dump()
        return await obesity_model_service.predict(input_data)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
