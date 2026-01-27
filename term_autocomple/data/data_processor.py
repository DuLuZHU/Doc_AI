#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据处理模块
负责医疗术语数据的加载、预处理和数据集构建
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from typing import List, Dict, Any, Tuple


class MedicalDataProcessor:
    """医疗数据处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据处理器
        
        Args:
            config: 配置字典，包含数据处理相关参数
        """
        self.config = config
        self.max_seq_length = config.get("max_seq_length", 128)
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        加载医疗术语数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载后的DataFrame数据
        """
        try:
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".json"):
                df = pd.read_json(file_path)
            elif file_path.endswith(".txt"):
                df = pd.read_csv(file_path, sep="\t")
            else:
                raise ValueError(f"不支持的文件格式: {file_path}")
            
            print(f"成功加载数据: {file_path}, 共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
    
    def preprocess_text(self, text: str) -> str:
        """
        文本预处理
        
        Args:
            text: 原始文本
            
        Returns:
            预处理后的文本
        """
        # 移除特殊字符
        text = text.strip()
        # 统一空格
        text = " ".join(text.split())
        # 转换为中文全角标点（可选）
        # 其他预处理步骤...
        return text
    
    def prepare_training_data(self, df: pd.DataFrame, text_col: str = "text") -> Dataset:
        """
        准备训练数据，转换为Hugging Face Dataset格式
        
        Args:
            df: 原始数据DataFrame
            text_col: 文本列名称
            
        Returns:
            处理后的Dataset对象
        """
        # 应用文本预处理
        df[text_col] = df[text_col].apply(self.preprocess_text)
        
        # 移除空值
        df = df.dropna(subset=[text_col])
        
        # 转换为Dataset
        dataset = Dataset.from_pandas(df)
        
        print(f"训练数据准备完成，共 {len(dataset)} 条有效记录")
        return dataset
    
    def create_masked_lm_data(self, dataset: Dataset, tokenizer) -> Dataset:
        """
        创建掩码语言模型训练数据
        
        Args:
            dataset: 原始数据集
            tokenizer: 分词器
            
        Returns:
            掩码语言模型训练数据集
        """
        def tokenize_function(examples):
            """分词函数"""
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_length
            )
        
        # 分词处理
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        print(f"掩码语言模型数据创建完成，共 {len(tokenized_dataset)} 条记录")
        return tokenized_dataset
    
    def split_dataset(self, dataset: Dataset, test_size: float = 0.1) -> Tuple[Dataset, Dataset]:
        """
        划分训练集和测试集
        
        Args:
            dataset: 完整数据集
            test_size: 测试集比例
            
        Returns:
            (训练集, 测试集) 元组
        """
        split_dataset = dataset.train_test_split(test_size=test_size, seed=42)
        train_dataset = split_dataset["train"]
        test_dataset = split_dataset["test"]
        
        print(f"数据集划分完成: 训练集 {len(train_dataset)} 条, 测试集 {len(test_dataset)} 条")
        return train_dataset, test_dataset
    
    def data_augmentation(self, texts: List[str], num_augments: int = 1) -> List[str]:
        """
        数据增强
        
        Args:
            texts: 原始文本列表
            num_augments: 每个文本增强的数量
            
        Returns:
            增强后的文本列表
        """
        augmented_texts = []
        
        for text in texts:
            # 原始文本保留
            augmented_texts.append(text)
            
            # 简单的数据增强：同义词替换（示例）
            # 实际应用中可以使用更复杂的增强方法
            for _ in range(num_augments):
                # 这里只是示例，实际需要实现具体的增强逻辑
                augmented_texts.append(text)  # 临时实现，返回原文本
        
        print(f"数据增强完成: 原始 {len(texts)} 条, 增强后 {len(augmented_texts)} 条")
        return augmented_texts
