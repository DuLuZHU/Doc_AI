#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数模块
提供通用的工具函数支持
"""

import os
import json
import logging
import time
import re
from typing import Dict, Any, List, Optional
import torch
import numpy as np


# ------------------------------
# 文件操作工具
# ------------------------------

def ensure_dir(directory: str) -> None:
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"创建目录: {directory}")

def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
        indent: 缩进空格数
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    
    print(f"数据已保存到: {file_path}")

def load_json(file_path: str) -> Any:
    """
    从JSON文件加载数据
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的数据
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data

def save_text(text: str, file_path: str) -> None:
    """
    保存文本到文件
    
    Args:
        text: 要保存的文本
        file_path: 文件路径
    """
    ensure_dir(os.path.dirname(file_path))
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    print(f"文本已保存到: {file_path}")

def load_text(file_path: str) -> str:
    """
    从文件加载文本
    
    Args:
        file_path: 文件路径
        
    Returns:
        加载的文本
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text


# ------------------------------
# 日志管理工具
# ------------------------------

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    设置日志配置
    
    Args:
        log_file: 日志文件路径，None表示只输出到控制台
        level: 日志级别
    """
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 根日志器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有处理器
    logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果提供了日志文件）
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    print(f"日志配置完成，日志级别: {logging.getLevelName(level)}")
    if log_file:
        print(f"日志文件: {log_file}")


# ------------------------------
# 时间工具
# ------------------------------

def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前时间
    
    Args:
        format: 时间格式
        
    Returns:
        格式化的当前时间字符串
    """
    return time.strftime(format, time.localtime())

def get_timestamp() -> str:
    """
    获取时间戳（用于文件名等）
    
    Returns:
        时间戳字符串
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

class Timer:
    """计时器类"""
    
    def __init__(self):
        self.start_time = None
    
    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
    
    def end(self) -> float:
        """
        结束计时并返回经过的时间
        
        Returns:
            经过的时间（秒）
        """
        if self.start_time is None:
            raise ValueError("计时器未启动")
        
        end_time = time.time()
        elapsed = end_time - self.start_time
        self.start_time = None
        
        return elapsed
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        elapsed = self.end()
        print(f"耗时: {elapsed:.4f}秒")


# ------------------------------
# 文本处理工具
# ------------------------------

def clean_text(text: str) -> str:
    """
    清理文本，去除特殊字符和多余空格
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 去除特殊字符，只保留中文、英文、数字和常用标点
    text = re.sub(r'[^一-龥a-zA-Z0-9，。！？；：、,.!?;: ]', '', text)
    
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip()
    
    return text

def split_sentences(text: str) -> List[str]:
    """
    将文本分割为句子
    
    Args:
        text: 原始文本
        
    Returns:
        句子列表
    """
    # 中文句子分割
    sentence_pattern = r'([。！？；：]|\n)'
    sentences = re.split(sentence_pattern, text)
    
    # 合并句子和分隔符
    result = []
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        if i + 1 < len(sentences):
            sentence += sentences[i + 1]
        
        sentence = sentence.strip()
        if sentence:
            result.append(sentence)
    
    return result

def extract_chinese(text: str) -> str:
    """
    提取文本中的中文
    
    Args:
        text: 原始文本
        
    Returns:
        只包含中文的文本
    """
    return ''.join(re.findall(r'[\u4e00-\u9fa5]', text))


# ------------------------------
# 模型相关工具
# ------------------------------

def count_model_parameters(model: torch.nn.Module) -> int:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        参数数量
    """
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    统计可训练参数数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        可训练参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    获取模型所在设备
    
    Args:
        model: PyTorch模型
        
    Returns:
        设备对象
    """
    return next(model.parameters()).device

def quantize_model(model: torch.nn.Module, dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
    """
    量化模型（动态量化）
    
    Args:
        model: PyTorch模型
        dtype: 量化数据类型
        
    Returns:
        量化后的模型
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
        dtype=dtype
    )
    
    print(f"模型量化完成，量化类型: {dtype}")
    return quantized_model


# ------------------------------
# 评估指标工具
# ------------------------------

def calculate_accuracy(predictions: List[Any], targets: List[Any]) -> float:
    """
    计算准确率
    
    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
        
    Returns:
        准确率
    """
    if len(predictions) != len(targets):
        raise ValueError("预测结果和真实标签长度不匹配")
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred == target)
    accuracy = correct / len(predictions)
    
    return accuracy

def calculate_precision_recall_f1(predictions: List[Any], targets: List[Any]) -> Dict[str, float]:
    """
    计算精确率、召回率和F1分数
    
    Args:
        predictions: 预测结果列表
        targets: 真实标签列表
        
    Returns:
        包含precision、recall、f1的字典
    """
    # 这里简化实现，假设是二分类问题
    # 实际应用中可能需要更复杂的实现
    tp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    fp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def calculate_mrr(ranked_predictions: List[List[Any]], targets: List[Any]) -> float:
    """
    计算平均倒数排名（Mean Reciprocal Rank）
    
    Args:
        ranked_predictions: 排序后的预测结果列表
        targets: 真实标签列表
        
    Returns:
        MRR值
    """
    reciprocal_ranks = []
    
    for preds, target in zip(ranked_predictions, targets):
        if target in preds:
            rank = preds.index(target) + 1  # 排名从1开始
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    return mrr

def calculate_hit_rate(ranked_predictions: List[List[Any]], targets: List[Any], k: int = 5) -> float:
    """
    计算Hit@K值
    
    Args:
        ranked_predictions: 排序后的预测结果列表
        targets: 真实标签列表
        k: 前K个结果
        
    Returns:
        Hit@K值
    """
    hits = []
    
    for preds, target in zip(ranked_predictions, targets):
        top_k_preds = preds[:k]
        hits.append(1.0 if target in top_k_preds else 0.0)
    
    hit_rate = np.mean(hits)
    return hit_rate


# ------------------------------
# 配置管理工具
# ------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    if config_path.endswith(".json"):
        return load_json(config_path)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except ImportError:
            raise ImportError("需要安装pyyaml库来加载YAML配置文件")
    else:
        raise ValueError(f"不支持的配置文件格式: {config_path}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并配置字典
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
