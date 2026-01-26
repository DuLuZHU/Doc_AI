# LLM API 医疗术语自动补全

使用大语言模型API（如DEEPSEEK）实现的医疗术语自动补全服务。

## 功能特点

- ✅ 无需训练，直接调用LLM API
- ✅ 支持带上下文的医疗术语补全
- ✅ 支持批量预测
- ✅ 灵活的配置管理
- ✅ 提供FastAPI接口
- ✅ 支持多种LLM模型

## 目录结构

```
llm_api/
├── config.py          # 配置文件
├── llm_predictor.py   # LLM API预测器
├── api.py             # FastAPI接口
├── example_usage.py   # 示例使用脚本
└── README.md          # 说明文档
```

## 快速开始

### 1. 配置API密钥

编辑 `config.py` 文件，设置你的LLM API密钥：

```python
llm_config = {
    "api_key": "your_actual_api_key_here",  # 替换为你的API密钥
    "model_name": "deepseek-chat",  # 模型名称
    "base_url": "http://192.167.253.100:33330/v1/chat/completions",  # API基础URL
    # ... 其他配置
}
```

### 2. 安装依赖

```bash
# 回到项目根目录
cd ..

# 安装依赖
pip install -r requirements.txt
```

### 3. 运行示例

```bash
# 进入llm_api目录
cd llm_api

# 运行示例脚本
python example_usage.py
```

### 4. 启动API服务

```bash
# 方式1：直接运行api.py
python api.py

# 方式2：使用uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5. 访问API文档

启动服务后，访问以下地址查看API文档：

```
http://localhost:8000/docs
```

## API接口

### 医疗术语自动补全

- **接口地址**：`POST /api/autocomplete`
- **请求参数**：
  - `query`：医生输入的查询词（如"高血"）
  - `context`：当前输入上下文（可选，如"患者有"）
  - `limit`：返回结果数量限制（默认5）

- **请求示例**：
  ```json
  {
    "query": "高血",
    "context": "患者有",
    "limit": 3
  }
  ```

- **响应示例**：
  ```json
  {
    "code": 200,
    "message": "success",
    "data": {
      "suggestions": [
        {
          "term": "高血压",
          "score": 1.0,
          "confidence": 1.0
        },
        {
          "term": "高血脂",
          "score": 2.0,
          "confidence": 0.75
        },
        {
          "term": "高血糖",
          "score": 3.0,
          "confidence": 0.5
        }
      ]
    }
  }
  ```

- **测试命令**：
  
  #### Linux/Mac (curl)
  ```bash
  curl -X POST "http://localhost:8000/api/autocomplete" -H "Content-Type: application/json" -d '{"query":"高血", "context":"患者有", "limit":3}'
  ```
  
  #### Windows PowerShell
  ```powershell
  Invoke-WebRequest -Uri "http://localhost:8000/api/autocomplete" -Method POST -ContentType "application/json" -Body '{"query":"高血", "context":"患者有", "limit":3}' -UseBasicParsing
  ```
  
  #### Windows PowerShell Core
  ```powershell
  curl.exe -X POST "http://localhost:8000/api/autocomplete" -H "Content-Type: application/json" -d '{"query":"高血", "context":"患者有", "limit":3}'
  ```

### 健康检查

- **接口地址**：`GET /health`
- **响应示例**：
  ```json
  {
    "code": 200,
    "message": "LLM API service is running",
    "timestamp": "2024-01-01T00:00:00Z"
  }
  ```

## 配置说明

### LLM API配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| api_key | str | your_api_key_here | API密钥 |
| model_name | str | deepseek-chat | 模型名称 |
| base_url | str | https://api.deepseek.com/v1/chat/completions | API基础URL |
| temperature | float | 0.1 | 生成温度 |
| max_tokens | int | 100 | 最大生成tokens |
| top_p | float | 0.9 | 核采样参数 |
| frequency_penalty | float | 0.0 | 频率惩罚 |
| presence_penalty | float | 0.0 | 存在惩罚 |
| timeout | int | 30 | 请求超时时间（秒） |
| retry_count | int | 3 | 重试次数 |

### 预测配置

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| top_k | int | 5 | 返回结果数量 |
| confidence_threshold | float | 0.3 | 置信度阈值 |
| max_length | int | 128 | 最大序列长度 |
| batch_size | int | 16 | 批量预测大小 |

### 提示词配置

可以在 `config.py` 中修改提示词，以适应不同的场景需求：

```python
prompt_config = {
    "system_prompt": "你是一个医疗术语自动补全助手。请根据医生输入的部分术语和上下文，补全完整的医疗术语。",
    "user_prompt_template": "上下文：{context}\n医生输入：{query}\n请补全完整的医疗术语，只返回最可能的{top_k}个结果，用逗号分隔。\n例如：\n上下文：患者有\n医生输入：高血\n返回：高血压,高血糖,高血脂\n\n现在请回答：",
}
```

## 支持的LLM模型

- [x] DeepSeek Chat
- [x] 可扩展支持其他LLM模型（如GPT、Claude、Gemini等）

## 使用不同的LLM模型

修改 `config.py` 中的 `base_url` 和 `model_name` 即可切换到其他LLM模型：

### 示例：使用OpenAI GPT

```python
llm_config = {
    "api_key": "your_openai_api_key",
    "model_name": "gpt-3.5-turbo",
    "base_url": "https://api.openai.com/v1/chat/completions",
    # ... 其他配置
}
```

### 示例：使用Claude

```python
llm_config = {
    "api_key": "your_anthropic_api_key",
    "model_name": "claude-3-sonnet-20240229",
    "base_url": "https://api.anthropic.com/v1/messages",
    # ... 其他配置
}
```

## 性能优化

- 调整 `temperature` 参数控制生成的多样性
- 调整 `timeout` 和 `retry_count` 参数处理API超时
- 使用 `batch_predict` 批量处理请求，减少API调用次数

## 注意事项

1. **API密钥安全**：请勿将API密钥提交到代码仓库
2. **API费用**：使用LLM API会产生费用，请合理控制调用次数
3. **结果过滤**：建议在客户端对结果进行二次过滤
4. **速率限制**：注意LLM API的速率限制，避免频繁调用
5. **医疗术语准确性**：建议对重要的医疗术语进行人工审核

## 许可证

MIT
