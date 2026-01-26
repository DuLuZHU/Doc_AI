#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件
集中管理训练和预测相关的配置参数
"""

from typing import Dict, Any


# ------------------------------
# 数据处理配置
# ------------------------------
data_config = {
    "max_seq_length": 128,  # 最大序列长度
    "test_size": 0.1,       # 测试集比例
    "text_column": "text",   # 文本列名称
    "mlm_probability": 0.15, # 掩码语言模型的掩码概率
    "data_augmentation": {
        "enabled": False,   # 是否启用数据增强
        "num_augments": 1   # 每个样本增强的数量
    }
}


# ------------------------------
# 模型训练配置
# ------------------------------
train_config = {
    "model_name": "model/bert-base-chinese",  # 预训练模型名称
    "learning_rate": 2e-5,               # 学习率
    "batch_size": 16,                    # 批量大小
    "num_epochs": 3,                     # 训练轮数
    "weight_decay": 0.01,                # 权重衰减
    "warmup_steps": 500,                 # 预热步数
    "logging_steps": 100,                # 日志记录步数
    "save_strategy": "epoch",           # 模型保存策略
    "save_total_limit": 3,               # 保存的模型总数限制
    "evaluation_strategy": "epoch",      # 评估策略
    "metric_for_best_model": "eval_loss", # 最佳模型的评估指标
    "load_best_model_at_end": True,      # 训练结束时加载最佳模型
    "report_to": "none",                 # 报告工具（none表示不使用）
    "fp16": False,                       # 是否使用混合精度训练
    "gradient_accumulation_steps": 1     # 梯度累积步数
}


# ------------------------------
# 模型预测配置
# ------------------------------
predict_config = {
    "top_k": 10,                   # 内部取top-k结果用于过滤
    "confidence_threshold": 0.3,    # 置信度阈值
    "max_length": 128,              # 最大序列长度
    "batch_size": 16,               # 批量预测大小
    "use_gpu": False                 # 是否使用GPU进行预测
}


# ------------------------------
# 路径配置
# ------------------------------
path_config = {
    "data_dir": "./data",                      # 数据目录
    "output_dir": "./output",                 # 输出目录
    "model_dir": "./medical_bert_model",      # 模型保存目录
    "log_dir": "./logs",                       # 日志目录
    "train_data_path": "./data/medical_train_data.csv",  # 训练数据路径
    "test_data_path": None,    # 测试数据路径
    "config_path": None            # 配置文件路径
}


# ------------------------------
# 完整配置字典
# ------------------------------
full_config = {
    "data": data_config,
    "train": train_config,
    "predict": predict_config,
    "path": path_config
}


def get_config(config_type: str = None) -> Dict[str, Any]:
    """
    获取配置
    
    Args:
        config_type: 配置类型，可选值："data", "train", "predict", "path"，
                    如果为None，则返回完整配置
        
    Returns:
        配置字典
    """
    if config_type is None:
        return full_config
    elif config_type in full_config:
        return full_config[config_type]
    else:
        raise ValueError(f"无效的配置类型: {config_type}，可选值: {list(full_config.keys())}")

def update_config(new_config: Dict[str, Any]) -> None:
    """
    更新配置
    
    Args:
        new_config: 新的配置字典，将覆盖原有配置
    """
    def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并字典"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    merge_dicts(full_config, new_config)
    print("配置已更新")

def load_config_from_file(file_path: str) -> None:
    """
    从文件加载配置
    
    Args:
        file_path: 配置文件路径
    """
    from utils import load_config as utils_load_config
    
    try:
        external_config = utils_load_config(file_path)
        update_config(external_config)
        print(f"配置已从文件加载: {file_path}")
    except Exception as e:
        print(f"从文件加载配置失败: {e}")
        raise


# ------------------------------
# 示例配置文件内容（JSON格式）
# ------------------------------
EXAMPLE_CONFIG_JSON = '''
{
  "data": {
    "max_seq_length": 128,
    "test_size": 0.1
  },
  "train": {
    "model_name": "bert-base-chinese",
    "learning_rate": 2e-5,
    "batch_size": 16,
    "num_epochs": 3
  },
  "predict": {
    "top_k": 10,
    "confidence_threshold": 0.3
  },
  "path": {
    "output_dir": "./output",
    "model_dir": "./medical_bert_model"
  }
}
'''
