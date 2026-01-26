#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM API 示例使用脚本
"""

from config import get_config
from llm_predictor import LLMModelPredictor


def main():
    """
    示例使用
    """
    print("=== LLM API 医疗术语自动补全示例 ===")
    
    # 1. 加载配置
    config = get_config()
    print("配置加载完成")
    
    # 2. 初始化预测器
    predictor = LLMModelPredictor(config)
    
    # 3. 示例预测
    print("\n--- 示例预测 ---\n")
    
    # 示例1：基本预测
    print("示例1：基本预测")
    result1 = predictor.predict("乳酸")
    print(f"查询：'乳酸'，结果：{[r['term'] for r in result1]}")
    
    # 示例2：带上下文的预测
    print("\n示例2：带上下文的预测")
    result2 = predictor.predict("心脏", "患者突发")
    print(f"查询：'心脏'，上下文：'患者突发'，结果：{[r['term'] for r in result2]}")
    
    # 示例3：限制结果数量
    print("\n示例3：限制结果数量")
    result3 = predictor.predict("糖尿", limit=3)
    print(f"查询：'糖尿'，限制返回3个结果：{[r['term'] for r in result3]}")
    
    # 示例4：批量预测
    print("\n示例4：批量预测")
    queries = ["心脏", "肝", "肺"]
    contexts = ["患者出现", "患者有", "患者感到"]
    results4 = predictor.batch_predict(queries, contexts)
    for query, context, result in zip(queries, contexts, results4):
        print(f"查询：'{query}'，上下文：'{context}'，结果：{[r['term'] for r in result]}")
    
    # 示例5：详细结果展示
    print("\n示例5：详细结果展示")
    result5 = predictor.predict("脑", "患者突发")
    for i, item in enumerate(result5, 1):
        print(f"  {i}. {item['term']} (置信度: {item['confidence']:.3f}, 分数: {item['score']:.2f})")
    
    print("\n=== 示例演示完成 ===")


if __name__ == "__main__":
    main()