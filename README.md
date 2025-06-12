# Utils 工具包

常用工具集合，便于日常数据处理、模型转换、平台适配等。

---

## 目录结构

```
tools/
├── cli_tools/                # 命令行直接调用工具
│   ├── image_concat.py       # 图像拼接
│   ├── unzips.py             # 批量解压
│   ├── copy_by_txt.py        # 按txt内容复制文件
├── func_utils/               # 仅函数调用型工具
│   ├── gen_logger.py         # 日志工具
│   ├── multi_process.py      # 多进程范本
│   └── ...                   # 其他函数型工具
├── dl_utils/                 # 深度学习相关工具
│   ├── ckpt_view.py          # ckpt参数查看
│   ├── pdparams2ckpt.py      # paddle转torch
│   ├── pth2ckpt.py           # torch转mindspore
│   └── pth2pdparams.py       # torch转paddle
├── openi_utils/              # 启智/平台相关
│   ├── openi.py              # obs数据迁移
│   ├── platform_process.py   # 多平台数据载入导出
│   ├── set_debug.py          # debug参数调整
│   └── set_environment.py    # 环境初始化
```

---

## 工具速查表

| 工具脚本                          | 功能简述                   | 适用平台         |
|-----------------------------------|----------------------------|------------------|
| [cli_tools/image_concat.py](tools/cli_tools/image_concat.py)         | 拼接两个文件夹下的图片      | paddle           |
| [cli_tools/unzips.py](tools/cli_tools/unzips.py)               | 批量解压zip                | all              |
| [cli_tools/copy_by_txt.py](tools/cli_tools/copy_by_txt.py)          | 按txt内容复制文件           | all              |
| [func_utils/gen_logger.py](tools/func_utils/gen_logger.py)          | 日志logger模板             | all              |
| [func_utils/multi_process.py](tools/func_utils/multi_process.py)        | 多进程运行模板              | all              |
| [dl_utils/ckpt_view.py](tools/dl_utils/ckpt_view.py)             | 查看ckpt参数名与内容        | mindspore        |
| [dl_utils/pdparams2ckpt.py](tools/dl_utils/pdparams2ckpt.py)         | paddle转torch参数           | paddle/torch     |
| [dl_utils/pth2ckpt.py](tools/dl_utils/pth2ckpt.py)               | torch转mindspore参数        | torch/mindspore  |
| [dl_utils/pth2pdparams.py](tools/dl_utils/pth2pdparams.py)           | torch转paddle参数           | paddle/torch     |
| [openi_utils/openi.py](tools/openi_utils/openi.py)              | obs与环境间数据迁移         | mindspore        |
| [openi_utils/platform_process.py](tools/openi_utils/platform_process.py)   | 多平台数据载入导出          | mindspore        |
| [openi_utils/set_debug.py](tools/openi_utils/set_debug.py)          | debug模式参数调整           | mindspore        |
| [openi_utils/set_environment.py](tools/openi_utils/set_environment.py)    | 环境初始化与混合精度        | mindspore        |

---

## 使用说明

- 详细参数和用法请查看对应脚本文件头部注释。
- 命令行工具已归类在`tools/cli_tools/`目录下。
- 仅函数调用型工具已归类在`tools/func_utils/`目录下。
- 深度学习相关工具已归类在`tools/dl_utils/`目录下。
- 平台相关（如openi、modelarts等）工具已归类在`tools/openi_utils/`目录下。

---

如需详细案例或参数说明，请直接查阅对应脚本文件。

