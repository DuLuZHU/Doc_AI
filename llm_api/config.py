#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM API 配置文件
集中管理LLM API调用相关的配置参数
"""

from typing import Dict, Any


# ------------------------------
# LLM API 配置
# ------------------------------
llm_config = {
    "api_key": "xxxxxxxxxxxxxxxxxxxxxxxx",  # API密钥
    "model_name": "deepseek-chat",  # 模型名称
    "base_url": "https://www.deepseek.cn",  # API基础URL
    "api_path": "/v1/chat/completions",  # API路径
    "temperature": 0.1,  # 生成温度
    "max_tokens": 100,  # 最大生成 tokens
    "top_p": 0.9,  # 核采样参数
    "frequency_penalty": 0.0,  # 频率惩罚
    "presence_penalty": 0.0,  # 存在惩罚
    "timeout": 30,  # 请求超时时间（秒）
    "retry_count": 3,  # 重试次数
    "verify_ssl": False,  # 是否验证SSL证书
    "disable_warnings": True,  # 是否禁用urllib3警告
}


# ------------------------------
# 预测配置
# ------------------------------
predict_config = {
    "top_k": 5,  # 返回结果数量
    "confidence_threshold": 0.3,  # 置信度阈值
    "max_length": 128,  # 最大序列长度
    "batch_size": 16,  # 批量预测大小
}


# ------------------------------
# 路径配置
# ------------------------------
path_config = {
    "output_dir": "./output",  # 输出目录
    "log_dir": "./logs",  # 日志目录
}


# ------------------------------
# 完整配置字典
# ------------------------------
full_config = {
    "llm": llm_config,
    "predict": predict_config,
    "path": path_config
}


# ------------------------------
# 提示词配置
# ------------------------------
prompt_config = {
    "system_prompt": "你是一个医疗术语自动补全助手。请根据医生输入的部分术语和上下文，补全完整的医疗术语。",
    "user_prompt_template": "上下文：{context}\n医生输入：{query}\n请补全完整的医疗术语，只返回最可能的{top_k}个结果，用逗号分隔，不要有其他内容。\n例如：\n上下文：患者有\n医生输入：高血\n返回：高血压,高血糖,高血脂\n\n现在请直接返回结果：",
}


def get_config(config_type: str = None) -> Dict[str, Any]:
    """
    获取配置
    
    Args:
        config_type: 配置类型，可选值："llm", "predict", "path", "prompt"，
                    如果为None，则返回完整配置
        
    Returns:
        配置字典
    """
    if config_type is None:
        return {
            **full_config,
            "prompt": prompt_config
        }
    elif config_type == "prompt":
        return prompt_config
    elif config_type in full_config:
        return full_config[config_type]
    else:
        raise ValueError(f"无效的配置类型: {config_type}，可选值: {list(full_config.keys()) + ['prompt']}")


def update_config(new_config: Dict[str, Any]) -> None:
    """
    更新配置
    
    Args:
        new_config: 新的配置字典，将覆盖原有配置
    """
    def merge_dicts(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并字典"""
        for key, value in update.items():
            if key == "prompt":
                base[key] = merge_dicts(prompt_config, value)
            elif key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
    
    merge_dicts(full_config, new_config)
    print("配置已更新")
