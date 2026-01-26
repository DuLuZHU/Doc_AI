# 医疗术语自动补全算法 - 使用说明

## 1. 算法概述

### 1.1 功能介绍
基于预训练语言模型的医疗术语自动补全算法，无需手动维护庞大的术语库，通过机器学习算法自动学习和预测医生输入意图，提供精准的自动补全建议。

### 1.2 核心优势
- **无需手动维护术语库**：模型自动学习和更新
- **支持上下文感知**：结合输入上下文提供更精准的建议
- **高扩展性**：轻松适应新术语和新领域
- **实时响应**：推理速度快，满足实时交互需求
- **可配置性强**：支持多种参数调整
- **易于集成**：提供简洁的API接口

### 1.3 应用场景
- 电子病历系统中的医生输入辅助
- 医疗术语查询系统
- 医疗文献写作辅助
- 医疗问答系统

## 2. 系统架构

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                     医疗术语自动补全算法                                            │
├───────────────────┬──────────────────┬────────────────────┬───────────┬───────────┤
│ 数据处理模块       │ 模型训练模块       │ 模型预测模块        │工具函数模块│ 配置文件   │
│ data_processor.py │ model_trainer.py │ model_predictor.py │ utils.py  │ config.py │
└───────────────────┴──────────────────┴────────────────────┴───────────┴───────────┘
         ▲                   ▲                   ▲
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────────────────────┐
                    │       训练数据文件      │
                    │ medical_train_data.csv │
                    └────────────────────────┘
```

### 2.1 模块说明

| 模块名称 | 文件路径 | 主要功能 |
|---------|---------|---------|
| 数据处理模块 | data_processor.py | 数据加载、预处理、数据集构建 |
| 模型训练模块 | model_trainer.py | 预训练模型加载、微调训练、模型保存 |
| 模型预测模块 | model_predictor.py | 模型加载、自动补全预测、批量预测 |
| 工具函数模块 | utils.py | 通用工具函数（文件操作、日志管理、文本处理等） |
| 配置文件 | config.py | 集中管理训练和预测的配置参数 |
| 示例训练数据 | medical_train_data.csv | 用于模型训练的示例医疗术语数据 |

## 3. 环境准备

### 3.1 硬件要求
- **CPU**：至少4核CPU
- **内存**：至少8GB RAM（加载BERT模型需要）
- **存储**：至少10GB可用空间（用于存储模型文件）
- **GPU**：可选，用于加速训练和推理

### 3.2 软件要求
- **Python**：3.8+
- **pip**：20.0+

### 3.3 依赖安装

创建并激活虚拟环境（可选但推荐）：

```bash
# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

安装依赖包：

```bash
pip install -r requirements.txt
```

## 4. 数据准备

### 4.1 数据格式

训练数据应为CSV格式，包含一列名为`text`的文本数据，每行一条医疗术语或相关上下文。示例：

```csv
text
患者有高血压病史
患者出现头痛症状
医生建议行血常规检查
患者诊断为2型糖尿病
给予药物治疗
```

### 4.2 示例数据

项目中已提供示例训练数据 `medical_train_data.csv`（由AI生成，数据质量较差，仅用于模型训练演示），包含了大量医疗术语和相关上下文，可以直接用于模型训练。

### 4.3 自定义数据

如果您想使用自己的医疗数据进行训练，只需按照上述格式准备CSV文件即可。

## 5. 模型训练

### 5.1 配置调整

在开始训练前，可以根据需要调整 `config.py` 中的配置参数：

```python
# 模型训练配置
train_config = {
    "model_name": "bert-base-chinese",  # 预训练模型名称
    "learning_rate": 2e-5,               # 学习率
    "batch_size": 16,                    # 批量大小
    "num_epochs": 3,                     # 训练轮数
    # 其他配置...
}

# 路径配置
path_config = {
    "train_data_path": "./medical_train_data.csv",  # 训练数据路径
    "model_dir": "./medical_bert_model",           # 模型保存目录
    # 其他配置...
}
```

### 5.2 执行训练

直接运行模型训练脚本：

```bash
python model_trainer.py
```

训练过程中，模型会定期保存到 `config.py` 中指定的 `model_dir` 目录。

### 5.3 训练过程说明

训练过程包括以下步骤：
1. 加载和预处理训练数据
2. 划分训练集和测试集
3. 加载预训练模型和分词器
4. 执行模型微调训练
5. 在测试集上评估模型
6. 保存训练好的模型

## 6. 模型预测

### 6.1 配置调整

在进行预测前，可以调整 `config.py` 中的预测相关配置：

```python
# 模型预测配置
predict_config = {
    "top_k": 10,                   # 内部取top-k结果用于过滤
    "confidence_threshold": 0.3,    # 置信度阈值
    "max_length": 128,              # 最大序列长度
    "use_gpu": True                 # 是否使用GPU进行预测
}

# 路径配置
path_config = {
    "model_dir": "./medical_bert_model",  # 模型路径
    # 其他配置...
}
```

### 6.2 执行预测

直接运行模型预测脚本：

```bash
python model_predictor.py
```

这将执行示例预测，展示算法的自动补全功能。

### 6.3 在Python代码中使用

您也可以在自己的Python代码中导入并使用 `MedicalModelPredictor` 类：

```python
from model_predictor import MedicalModelPredictor

# 配置
config = {
    "model_path": "./medical_bert_model",
    "top_k": 10,
    "confidence_threshold": 0.3
}

# 初始化预测器
predictor = MedicalModelPredictor(config)

# 执行预测
results = predictor.predict(query="高血", context="患者有", limit=5)

# 打印结果
for i, result in enumerate(results, 1):
    print(f"{i}. {result['term']} (分数: {result['score']:.4f}, 置信度: {result['confidence']:.3f})")
```

### 6.4 批量预测

支持批量预测，提高预测效率：

```python
# 批量预测示例
queries = ["头", "肺", "心", "胃"]
contexts = ["患者出现", "患者有", "医生诊断为", "患者感到"]
results = predictor.batch_predict(queries, contexts, limit=3)

for query, context, res in zip(queries, contexts, results):
    print(f"\n查询: '{query}', 上下文: '{context}'")
    for i, item in enumerate(res, 1):
        print(f"{i}. {item['term']} (置信度: {item['confidence']:.3f})")
```

## 7. 配置详解

### 7.1 数据处理配置

| 参数名 | 默认值 | 说明 |
|-------|-------|------|
| max_seq_length | 128 | 最大序列长度，超过该长度的文本将被截断 |
| test_size | 0.1 | 测试集比例，用于划分训练集和测试集 |
| text_column | "text" | 训练数据中包含文本的列名 |
| mlm_probability | 0.15 | 掩码语言模型的掩码概率 |
| data_augmentation.enabled | False | 是否启用数据增强 |
| data_augmentation.num_augments | 1 | 每个样本增强的数量 |

### 7.2 模型训练配置

| 参数名 | 默认值 | 说明 |
|-------|-------|------|
| model_name | "bert-base-chinese" | 预训练模型名称，可替换为医疗领域预训练模型 |
| learning_rate | 2e-5 | 学习率 |
| batch_size | 16 | 批量大小，根据可用内存调整 |
| num_epochs | 3 | 训练轮数 |
| weight_decay | 0.01 | 权重衰减，用于防止过拟合 |
| warmup_steps | 500 | 预热步数，学习率从0逐渐增加到设定值 |
| logging_steps | 100 | 日志记录步数，每训练多少步记录一次日志 |
| save_steps | 500 | 模型保存步数，每训练多少步保存一次模型 |
| save_total_limit | 3 | 保存的模型总数限制，超过该数量会删除旧模型 |
| evaluation_strategy | "epoch" | 评估策略，可选值："epoch"（每轮评估一次）、"steps"（每步评估一次）、"no"（不评估） |
| metric_for_best_model | "eval_loss" | 选择最佳模型的评估指标 |
| load_best_model_at_end | True | 训练结束时是否加载最佳模型 |
| fp16 | False | 是否使用混合精度训练，可加速训练并减少内存使用 |

### 7.3 模型预测配置

| 参数名 | 默认值 | 说明 |
|-------|-------|------|
| top_k | 10 | 内部取top-k结果用于过滤 |
| confidence_threshold | 0.3 | 置信度阈值，低于该值的结果将被过滤 |
| max_length | 128 | 最大序列长度 |
| batch_size | 32 | 批量预测大小 |
| use_gpu | True | 是否使用GPU进行预测 |

### 7.4 路径配置

| 参数名 | 默认值 | 说明 |
|-------|-------|------|
| data_dir | "./data" | 数据目录 |
| output_dir | "./output" | 输出目录 |
| model_dir | "./medical_bert_model" | 模型保存目录 |
| log_dir | "./logs" | 日志目录 |
| train_data_path | "./medical_train_data.csv" | 训练数据路径 |
| test_data_path | "./medical_test_data.csv" | 测试数据路径 |
| config_path | "./config.json" | 配置文件路径 |

## 8. 常见问题与解决方案

### 8.1 模型下载失败

**问题**：首次训练时，预训练模型下载失败。

**解决方案**：
- 检查网络连接
- 手动下载模型文件并放置到指定目录
- 设置代理：
  ```bash
  export HTTP_PROXY=http://proxy.example.com:8080
  export HTTPS_PROXY=http://proxy.example.com:8080
  ```

### 8.2 内存不足

**问题**：训练或预测时出现内存不足错误。

**解决方案**：
- 减少 `batch_size` 参数
- 使用更小的预训练模型（如 `distilbert-base-chinese`）
- 启用 `fp16` 混合精度训练
- 增加系统内存

### 8.3 训练速度慢

**解决方案**：
- 使用GPU进行训练
- 增加 `batch_size`（如果内存允许）
- 使用更小的预训练模型
- 启用 `fp16` 混合精度训练

### 8.4 预测结果不准确

**解决方案**：
- 增加训练轮数
- 使用更大的预训练模型
- 准备更多高质量的训练数据
- 调整 `learning_rate` 等训练参数
- 调整预测时的 `confidence_threshold` 参数

### 8.5 中文显示问题

**解决方案**：
- 确保所有代码文件使用UTF-8编码
- 确保终端支持中文显示

## 9. 扩展与优化

### 9.1 使用医疗领域预训练模型

建议使用医疗领域预训练模型以获得更好的效果，如：
- 哈工大医疗BERT
- 百度ERNIE Health
- 阿里云医疗大模型

只需在 `config.py` 中修改 `model_name` 参数即可：

```python
train_config = {
    "model_name": "哈工大/医疗BERT",  # 替换为医疗领域模型名称
    # 其他配置...
}
```

### 9.2 模型量化

可以对训练好的模型进行量化，以减少内存占用和提高推理速度：

```python
from utils import quantize_model

# 加载模型
model = AutoModelForMaskedLM.from_pretrained("./medical_bert_model")

# 量化模型
quantized_model = quantize_model(model)

# 保存量化模型
quantized_model.save_pretrained("./quantized_medical_model")
```

### 9.3 模型部署

可以将训练好的模型部署为API服务，方便集成到现有系统：

```python
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
```

启动API服务：
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

API服务启动后，打开浏览器，访问以下地址：

```
http://localhost:8000/docs
```

这将打开Swagger UI，您可以在其中测试API接口。

---

本项目绝大部分的代码由Trae生成。

通过本使用说明文档，您可以快速上手使用这个算法，并根据需要进行调整和扩展。如果您在使用过程中遇到任何问题，欢迎随时反馈和交流。

---

**版本**：v1.0.0(demo)

**更新日期**：2026-01-04  

**开发者**：Mengjing Zhu