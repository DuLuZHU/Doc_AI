#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM API 医疗术语自动补全包
"""

from .config import get_config, update_config
from .llm_predictor import LLMModelPredictor


__version__ = "1.0.0"
__all__ = [
    "get_config",
    "update_config",
    "LLMModelPredictor"
]