#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型预测模块
负责使用训练好的模型进行医疗术语自动补全预测
"""

import sys
import os

# 将项目根目录添加到sys.path，确保能够正确导入模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Any, Optional


class MedicalModelPredictor:
    """医疗模型预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型预测器
        
        Args:
            config: 配置字典，包含模型预测相关参数
        """
        self.config = config
        
        # 提取配置参数
        self.predict_config = config.get("predict", {})
        self.path_config = config.get("path", {})
        
        # 模型路径
        self.model_path = self.path_config.get("model_dir", "./medical_bert_model")
        
        # 预测参数
        self.top_k = self.predict_config.get("top_k", 10)
        self.confidence_threshold = self.predict_config.get("confidence_threshold", 0.3)
        self.max_length = self.predict_config.get("max_length", 128)
        self.batch_size = self.predict_config.get("batch_size", 16)
        self.use_gpu = self.predict_config.get("use_gpu", False)
        
        # 设备选择
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.tokenizer = None
        self.model = None
        
        # 初始化时加载模型
        self.load_model()
    
    def load_model(self) -> None:
        """
        加载训练好的模型和分词器
        """
        try:
            print(f"正在加载模型: {self.model_path}")
            print(f"使用设备: {self.device}")
            
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # 加载模型
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"成功加载模型: {self.model_path}")
            
        except Exception as e:
            print(f"加载模型失败: {e}")
            raise
    
    def predict(self, query: str, context: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """
        执行自动补全预测
        
        Args:
            query: 医生输入的查询词（如"高血"）
            context: 当前输入上下文（如"患者有"）
            limit: 返回结果数量限制
            
        Returns:
            预测结果列表，包含术语、分数、置信度等
        """
        if not query:
            return []
        
        try:
            # 构造输入文本
            if context:
                input_text = f"{context} {query}[MASK]"
            else:
                input_text = f"{query}[MASK]"
            
            # 编码输入
            inputs = self.tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # 找到MASK位置
            mask_positions = (inputs["input_ids"] == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            # 确保只处理第一个MASK标记
            if len(mask_positions) == 0:
                print("未找到MASK标记")
                return []
            
            mask_token_index = mask_positions[0]
            mask_logits = logits[0, mask_token_index, :]
            
            # 获取top-k预测
            top_k_results = mask_logits.topk(min(self.top_k, 30))  # 取更多结果用于过滤
            top_k_indices = top_k_results.indices.tolist()
            top_k_scores = top_k_results.values.tolist()
            
            # 解码并过滤结果
            suggestions = []
            seen_terms = set()
            
            for token_id, score in zip(top_k_indices, top_k_scores):
                # 解码预测的token
                predicted_token = self.tokenizer.decode([token_id])
                full_term = f"{query}{predicted_token}"
                
                # 过滤规则
                if self._is_valid_term(full_term, seen_terms):
                    # 计算置信度（使用sigmoid归一化）
                    confidence = torch.sigmoid(torch.tensor(score)).item()
                    
                    # 只保留置信度高于阈值的结果
                    if confidence >= self.confidence_threshold:
                        suggestions.append({
                            "term": full_term,
                            "score": float(score),
                            "confidence": round(confidence, 3)
                        })
                        
                        seen_terms.add(full_term)
                        
                        # 如果已经收集了足够的结果，提前退出
                        if len(suggestions) >= limit:
                            break
            
            # 按分数排序
            suggestions.sort(key=lambda x: x["score"], reverse=True)
            
            return suggestions[:limit]
            
        except Exception as e:
            print(f"预测失败: {e}")
            return []
    
    def _is_valid_term(self, term: str, seen_terms: set) -> bool:
        """
        验证术语是否有效
        
        Args:
            term: 术语
            seen_terms: 已见过的术语集合
            
        Returns:
            是否有效的布尔值
        """
        # 避免重复
        if term in seen_terms:
            return False
        
        # 长度检查
        if not (2 <= len(term) <= 20):
            return False
        
        # 特殊字符检查
        if any(char in term for char in ["[", "]", "<", ">", "_", "^", "$", "*", "&", "#"]):
            return False
        
        # 只包含空格的检查
        if term.strip() == "":
            return False
        
        return True
    
    def batch_predict(self, queries: List[str], contexts: Optional[List[str]] = None, limit: int = 5) -> List[List[Dict[str, Any]]]:
        """
        批量预测
        
        Args:
            queries: 查询词列表
            contexts: 上下文列表，与查询词一一对应
            limit: 每个查询返回的结果数量限制
            
        Returns:
            批量预测结果列表
        """
        if contexts is None:
            contexts = [""] * len(queries)
        
        results = []
        for query, context in zip(queries, contexts):
            result = self.predict(query, context, limit)
            results.append(result)
        
        return results


def main():
    """
    主函数，用于执行模型预测
    """
    from config import get_config
    
    print("=== 医疗模型预测 ===")
    
    # 1. 加载配置
    config = get_config()
    print(f"配置加载完成")
    
    try:
        # 2. 初始化预测器
        predictor = MedicalModelPredictor(config)
        
        # 3. 示例预测
        example_queries = ["高血", "糖尿", "心脏"]
        example_contexts = ["患者有", "患者诊断为", "患者出现"]
        
        print("\n示例预测结果：")
        for query, context in zip(example_queries, example_contexts):
            print(f"\n查询: '{query}'，上下文: '{context}'")
            results = predictor.predict(query, context, limit=3)
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['term']} (置信度: {result['confidence']:.3f})")
    except Exception as e:
        print(f"预测执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()