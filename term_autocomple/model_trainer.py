#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练模块
负责医疗术语自动补全模型的加载、微调训练和保存
"""

import sys
import os

# 将项目根目录添加到sys.path，确保能够正确导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from typing import Dict, Any, Optional

# 导入数据处理器
from data.data_processor import MedicalDataProcessor


class MedicalModelTrainer:
    """医疗模型训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型训练器
        
        Args:
            config: 配置字典，包含完整的配置信息
        """
        self.config = config
        
        # 提取配置参数
        self.data_config = config.get("data", {})
        self.train_config = config.get("train", {})
        self.path_config = config.get("path", {})
        
        # 模型和路径配置
        self.model_name = self.train_config.get("model_name", "bert-base-chinese")
        self.output_dir = self.path_config.get("model_dir", "./medical_bert_model")
        self.data_dir = self.path_config.get("data_dir", "./data")
        self.train_data_path = self.path_config.get("train_data_path", "./data/medical_train_data.csv")
        self.test_data_path = self.path_config.get("test_data_path", None)
        
        # 训练参数
        self.learning_rate = self.train_config.get("learning_rate", 2e-5)
        self.batch_size = self.train_config.get("batch_size", 16)
        self.num_epochs = self.train_config.get("num_epochs", 3)
        self.weight_decay = self.train_config.get("weight_decay", 0.01)
        self.warmup_steps = self.train_config.get("warmup_steps", 500)
        self.logging_steps = self.train_config.get("logging_steps", 100)
        self.save_strategy = self.train_config.get("save_strategy", "epoch")
        self.save_total_limit = self.train_config.get("save_total_limit", 3)
        self.evaluation_strategy = self.train_config.get("evaluation_strategy", "epoch")
        self.metric_for_best_model = self.train_config.get("metric_for_best_model", "eval_loss")
        self.load_best_model_at_end = self.train_config.get("load_best_model_at_end", True)
        self.report_to = self.train_config.get("report_to", "none")
        self.fp16 = self.train_config.get("fp16", False)
        self.gradient_accumulation_steps = self.train_config.get("gradient_accumulation_steps", 1)
        
        # 数据处理参数
        self.max_seq_length = self.data_config.get("max_seq_length", 128)
        self.test_size = self.data_config.get("test_size", 0.1)
        self.text_column = self.data_config.get("text_column", "text")
        self.mlm_probability = self.data_config.get("mlm_probability", 0.15)
        self.data_augmentation = self.data_config.get("data_augmentation", {})
        
        # 初始化数据处理器
        self.data_processor = MedicalDataProcessor(self.data_config)
        
        self.tokenizer = None
        self.model = None
    
    def load_model(self) -> None:
        """
        加载预训练模型和分词器
        """
        try:
            print(f"正在加载预训练模型: {self.model_name}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 加载模型
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            
            print(f"成功加载模型: {self.model_name}")
            print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None) -> None:
        """
        执行模型训练
        
        Args:
            train_dataset: 训练数据集
            eval_dataset: 评估数据集（可选）
        """
        if not self.tokenizer or not self.model:
            self.load_model()
        
        # 创建数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )
        
        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy=self.evaluation_strategy if eval_dataset else "no",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            logging_steps=self.logging_steps,
            save_strategy=self.save_strategy,
            save_total_limit=self.save_total_limit,
            load_best_model_at_end=self.load_best_model_at_end if eval_dataset else False,
            metric_for_best_model=self.metric_for_best_model if eval_dataset else None,
            report_to=self.report_to,
            fp16=self.fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        print("开始模型训练...")
        trainer.train()
        
        print("模型训练完成!")
    
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            eval_dataset: 评估数据集
            
        Returns:
            评估指标字典
        """
        if not self.tokenizer or not self.model:
            self.load_model()
        
        # 创建数据收集器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )
        
        # 初始化Trainer
        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 执行评估
        print("开始模型评估...")
        eval_results = trainer.evaluate(eval_dataset)
        
        print(f"评估完成: {eval_results}")
        return eval_results
    
    def save_model(self, save_path: Optional[str] = None) -> None:
        """
        保存训练好的模型
        
        Args:
            save_path: 模型保存路径，默认使用配置中的output_dir
        """
        if not self.tokenizer or not self.model:
            raise ValueError("模型未加载或未训练")
        
        save_dir = save_path or self.output_dir
        
        print(f"正在保存模型到: {save_dir}")
        
        # 保存模型和分词器
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存配置
        import json
        with open(f"{save_dir}/training_config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        
        print(f"模型保存成功: {save_dir}")
    
    def load_trained_model(self, model_path: str) -> None:
        """
        加载已训练好的模型
        
        Args:
            model_path: 模型路径
        """
        try:
            print(f"正在加载已训练模型: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForMaskedLM.from_pretrained(model_path)
            
            print(f"成功加载已训练模型: {model_path}")
            
        except Exception as e:
            print(f"加载已训练模型失败: {e}")
            raise
    
    def load_data(self, data_path: str) -> Dataset:
        """
        加载数据集
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            加载的数据集
        """
        # 使用数据处理器加载数据
        df = self.data_processor.load_data(data_path)
        # 准备训练数据
        dataset = self.data_processor.prepare_training_data(df, self.text_column)
        return dataset
    
    def preprocess_data(self, dataset: Dataset) -> Dataset:
        """
        预处理数据集
        
        Args:
            dataset: 原始数据集
            
        Returns:
            预处理后的数据集
        """
        if not self.tokenizer:
            self.load_model()
        
        # 使用数据处理器创建掩码语言模型数据
        processed_dataset = self.data_processor.create_masked_lm_data(dataset, self.tokenizer)
        return processed_dataset
    
    def split_dataset(self, dataset: Dataset) -> tuple[Dataset, Optional[Dataset]]:
        """
        分割数据集为训练集和评估集
        
        Args:
            dataset: 完整数据集
            
        Returns:
            训练集和评估集
        """
        print(f"正在分割数据集，测试集比例: {self.test_size}")
        
        if self.test_data_path:
            # 使用独立的测试集
            eval_dataset = self.load_data(self.test_data_path)
            train_dataset = dataset
        else:
            # 使用数据处理器分割数据集
            train_dataset, eval_dataset = self.data_processor.split_dataset(dataset, self.test_size)
        
        return train_dataset, eval_dataset
    
    def run_training(self) -> None:
        """
        完整的训练流程：加载数据 -> 预处理 -> 分割 -> 训练 -> 评估 -> 保存模型
        """
        print("=== 开始完整训练流程 ===")
        
        # 1. 加载数据
        train_data = self.load_data(self.train_data_path)
        
        # 2. 预处理数据
        processed_data = self.preprocess_data(train_data)
        
        # 3. 分割数据集
        train_dataset, eval_dataset = self.split_dataset(processed_data)
        
        # 4. 加载模型
        self.load_model()
        
        # 5. 训练模型
        self.train(train_dataset, eval_dataset)
        
        # 6. 评估模型
        if eval_dataset:
            self.evaluate(eval_dataset)
        
        # 7. 保存模型
        self.save_model()
        
        print("=== 训练流程完成 ===")


def main():
    """
    主函数，用于执行模型训练
    """
    from config import get_config
    
    print("=== 医疗模型训练 ===")
    
    # 1. 加载配置
    config = get_config()
    print(f"配置加载完成: 使用模型 {config['train']['model_name']}")
    
    # 2. 初始化训练器
    trainer = MedicalModelTrainer(config)
    
    # 3. 执行训练
    trainer.run_training()


if __name__ == "__main__":
    main()