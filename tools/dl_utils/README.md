# dl_utils

深度学习权重格式转换与相关工具包

## 功能简介

本目录下包含多种主流深度学习框架（PyTorch、PaddlePaddle、MindSpore）之间的权重转换工具，以及权重查看、参数适配等辅助脚本，适用于模型迁移、兼容、复现等场景。

## 工具列表

| 脚本名                | 功能描述                       | 典型用法示例 |
|----------------------|------------------------------|-------------|
| model_param_converter.py | 通用权重格式转换（支持三大主流框架互转） | python model_param_converter.py --config your_config.json |
| pth2ckpt.py          | PyTorch权重转MindSpore权重     | python pth2ckpt.py --pth xxx.pth --ckpt xxx.ckpt |
| pdparams2ckpt.py     | PaddlePaddle权重转MindSpore权重 | python pdparams2ckpt.py --pdparams xxx.pdparams --ms-model-fn ... |
| pth2pdparams.py      | PyTorch权重转PaddlePaddle权重   | python pth2pdparams.py --pth xxx.pth --pdparams xxx.pdparams |
| ckpt_view.py         | MindSpore权重内容查看           | python ckpt_view.py --ckpt xxx.ckpt |

## 使用说明

- 推荐优先使用 model_param_converter.py 及其 json 配置文件进行权重转换，支持灵活参数配置。
- 详细参数和用法请查阅各脚本头部注释。
- 适合模型迁移、参数兼容、深度学习实验等场景。

## 依赖

- Python 3.6+
- torch、paddlepaddle、mindspore、numpy 等（按需安装）

---

如需定制权重转换规则或支持新框架，请参考 model_param_converter.py 进行扩展。

