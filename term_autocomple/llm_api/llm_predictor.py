#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM API 预测模块
负责使用大语言模型API进行医疗术语自动补全预测
"""

import sys
import os
import requests
import time
from typing import List, Dict, Any, Optional


class LLMModelPredictor:
    """LLM模型预测器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化LLM模型预测器
        
        Args:
            config: 配置字典，包含LLM API调用相关参数
        """
        self.config = config
        
        # 提取配置参数
        self.llm_config = config.get("llm", {})
        self.predict_config = config.get("predict", {})
        self.prompt_config = config.get("prompt", {})
        
        # LLM API参数
        self.api_key = self.llm_config.get("api_key")
        self.model_name = self.llm_config.get("model_name", "deepseek-chat")
        self.base_url = self.llm_config.get("base_url", "https://api.deepseek.com")
        self.api_path = self.llm_config.get("api_path", "/v1/chat/completions")
        self.temperature = self.llm_config.get("temperature", 0.1)
        self.max_tokens = self.llm_config.get("max_tokens", 100)
        self.timeout = self.llm_config.get("timeout", 30)
        self.retry_count = self.llm_config.get("retry_count", 3)
        self.verify_ssl = self.llm_config.get("verify_ssl", False)
        self.disable_warnings = self.llm_config.get("disable_warnings", True)
        
        # 预测参数
        self.top_k = self.predict_config.get("top_k", 5)
        self.confidence_threshold = self.predict_config.get("confidence_threshold", 0.3)
        
        # 提示词配置
        self.system_prompt = self.prompt_config.get("system_prompt", "你是一个医疗术语自动补全助手。")
        self.user_prompt_template = self.prompt_config.get("user_prompt_template", "上下文：{context}\n医生输入：{query}\n请补全完整的医疗术语，只返回最可能的{top_k}个结果，用逗号分隔。")
        
        # 检查API密钥
        if not self.api_key or self.api_key == "your_api_key_here":
            print("警告：未配置有效的API密钥，请在config.py中设置api_key")
        
        # 禁用urllib3警告
        if self.disable_warnings:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def _call_llm_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        调用LLM API
        
        Args:
            messages: 消息列表
            
        Returns:
            API返回结果
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.llm_config.get("top_p", 0.9),
            "frequency_penalty": self.llm_config.get("frequency_penalty", 0.0),
            "presence_penalty": self.llm_config.get("presence_penalty", 0.0),
        }
        
        for attempt in range(self.retry_count):
            try:
                # 构造完整的API URL
                full_url = f"{self.base_url.rstrip('/')}/{self.api_path.lstrip('/')}"
                
                response = requests.post(
                    full_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                    verify=self.verify_ssl
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"LLM API调用失败（尝试 {attempt+1}/{self.retry_count}）: {e}")
                # 打印完整的错误响应
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"错误响应内容: {e.response.text}")
                except:
                    pass
                
                if attempt < self.retry_count - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise
    
    def _generate_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """
        生成提示词
        
        Args:
            query: 医生输入的查询词
            context: 当前输入上下文
            
        Returns:
            格式化的消息列表
        """
        user_prompt = self.user_prompt_template.format(
            context=context,
            query=query,
            top_k=self.top_k
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return messages
    
    def _parse_response(self, response: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """
        解析LLM API响应
        
        Args:
            response: API返回结果
            query: 医生输入的查询词
            
        Returns:
            格式化的预测结果列表
        """
        try:
            # 提取生成的文本
            generated_text = response["choices"][0]["message"]["content"].strip()
            
            # 解析结果（假设返回格式为：术语1,术语2,术语3）
            terms = [term.strip() for term in generated_text.split(",") if term.strip()]
            
            # 过滤结果
            filtered_terms = []
            seen_terms = set()
            
            for i, term in enumerate(terms):
                # 跳过重复项
                if term in seen_terms:
                    continue
                
                # 确保术语以查询词开头
                if not term.startswith(query):
                    continue
                
                # 计算置信度（基于返回顺序，简单实现）
                confidence = 1.0 - (i / len(terms)) * 0.5
                
                # 只保留置信度高于阈值的结果
                if confidence >= self.confidence_threshold:
                    filtered_terms.append({
                        "term": term,
                        "score": float(len(filtered_terms) + 1),  # 简单评分
                        "confidence": round(confidence, 3)
                    })
                    seen_terms.add(term)
                
                # 达到top_k数量则停止
                if len(filtered_terms) >= self.top_k:
                    break
            
            return filtered_terms
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            return []
    
    def predict(self, query: str, context: str = "", limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
            # 使用指定的limit或默认值
            current_top_k = limit if limit is not None else self.top_k
            
            # 生成提示词
            messages = self._generate_prompt(query, context)
            
            # 调用LLM API
            response = self._call_llm_api(messages)
            
            # 解析响应
            results = self._parse_response(response, query)
            
            # 按置信度排序并返回指定数量
            results.sort(key=lambda x: x["confidence"], reverse=True)
            return results[:current_top_k]
            
        except Exception as e:
            print(f"预测失败: {e}")
            return []
    
    def batch_predict(self, queries: List[str], contexts: Optional[List[str]] = None, limit: Optional[int] = None) -> List[List[Dict[str, Any]]]:
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
