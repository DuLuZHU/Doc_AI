from fastapi import FastAPI
from pydantic import BaseModel
from model_predictor import MedicalModelPredictor

app = FastAPI()

# 初始化预测器
predictor = MedicalModelPredictor({
    "model_path": "./medical_bert_model"
})

# 请求模型
class AutocompleteRequest(BaseModel):
    query: str
    context: str = ""
    limit: int = 5

@app.post("/api/autocomplete")
def autocomplete(request: AutocompleteRequest):
    results = predictor.predict(
        query=request.query,
        context=request.context,
        limit=request.limit
    )
    return {
        "code": 200,
        "message": "success",
        "data": {"suggestions": results}
    }