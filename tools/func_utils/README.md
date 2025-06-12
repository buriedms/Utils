# func_utils

通用函数与工具集合

## 功能简介

本目录下收录了常用的 Python 函数工具，适用于日志管理、多进程处理等通用场景，便于在各类项目中复用。

## 工具列表

| 脚本名           | 功能描述             | 典型用法示例 |
|------------------|----------------------|-------------|
| gen_logger.py    | 日志logger生成与管理 | from func_utils.gen_logger import gen_logger |
| multi_process.py | 多进程辅助函数       | from func_utils.multi_process import ... |

## 使用说明

- 直接 import 需要的函数或类到你的项目中。
- 适合需要日志、并发等通用能力的脚本或工程。
- 详细参数和用法请查阅各脚本头部注释。

## 依赖

- Python 3.6+
- 可能依赖 logging、multiprocessing 等标准库。

---

如需扩展更多通用函数，可在本目录下新增脚本并完善注释。

