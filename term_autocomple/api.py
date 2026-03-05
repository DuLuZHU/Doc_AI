from fastapi import FastAPI
from pydantic import BaseModel
from model_predictor import MedicalModelPredictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="医疗术语自动补全API (BERT)",
    description="使用Transformer模型实现的医疗术语自动补全服务",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def set_encoding_middleware(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

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