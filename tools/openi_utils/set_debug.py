"""
调试模式配置工具

功能：
- 一键切换调试模式，自动缩小batch、输入尺寸、训练轮数等。
- 自动切换本地数据集路径，便于本地调试。
- 支持通过debug_json_path参数读取外部json文件，递归覆写config，实现调试参数的灵活配置，无需修改源代码。

依赖：os, json

参数说明：
- config: 需包含 debug 字段及 config.dataset、config.train 等子字段。
- debug_json_path: 可选，外部json路径，优先级高于硬编码。

用法：
    from set_debug import set_debug
    set_debug(config, debug_json_path="path/to/debug.json")
"""
import os
import json

def set_debug(config, debug_json_path="./default_debug.json"):
    """
    启用调试模式，自动缩小数据与训练规模，切换本地数据路径。
    仅通过debug_json_path读取外部json进行参数覆写。
    :param config: 配置对象，需包含 debug、dataset、train 等字段
    :param debug_json_path: 外部json路径，默认'./default_debug.json'
    """
    if config.debug:
        if debug_json_path and os.path.exists(debug_json_path):
            with open(debug_json_path, 'r', encoding='utf-8') as f:
                debug_cfg = json.load(f)
            def update_obj(obj, update_dict):
                for k, v in update_dict.items():
                    if hasattr(obj, k):
                        if isinstance(v, dict):
                            update_obj(getattr(obj, k), v)
                        else:
                            setattr(obj, k, v)
            update_obj(config, debug_cfg)
        else:
            raise FileNotFoundError(f"未找到调试参数json文件: {debug_json_path}")
