#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM API 接口实现
使用FastAPI提供医疗术语自动补全API
"""

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 处理相对导入问题，支持两种运行模式
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config
from llm_predictor import LLMModelPredictor

app = FastAPI(
    title="医疗术语自动补全API (LLM)",
    description="使用大语言模型API实现的医疗术语自动补全服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置默认的JSON响应编码
@app.middleware("http")
async def set_encoding_middleware(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response

# 初始化预测器
config = get_config()
predictor = LLMModelPredictor(config)


# 请求模型
class AutocompleteRequest(BaseModel):
    query: str
    context: str = ""
    limit: int = 5


@app.post("/api/autocomplete", tags=["医疗术语自动补全"])
def autocomplete(request: AutocompleteRequest):
    """
    医疗术语自动补全
    
    Args:
        query: 医生输入的查询词（如"高血"）
        context: 当前输入上下文（可选，如"患者有"）
        limit: 返回结果数量限制（默认5）
        
    Returns:
        预测结果列表
    """
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


@app.get("/health", tags=["健康检查"])
def health_check():
    """
    健康检查接口
    
    Returns:
        服务状态
    """
    return {
        "code": 200,
        "message": "LLM API service is running",
        "timestamp": "2024-01-01T00:00:00Z"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)