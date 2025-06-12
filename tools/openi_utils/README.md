# openi_utils

OpenI/ModelArts 平台环境、数据、调试等自动化工具包

## 功能简介

本目录下包含适用于启智OpenI、ModelArts等平台的环境初始化、数据同步、调试参数管理等自动化脚本，便于云端/本地训练、数据迁移、分布式实验等场景。

## 工具列表

| 脚本名                | 功能描述                       | 典型用法示例 |
|----------------------|------------------------------|-------------|
| set_environment.py   | MindSpore环境与分布式初始化    | from openi_utils import set_environment; set_environment(config) |
| set_debug.py         | 调试参数自动覆写与本地调试      | from openi_utils import set_debug; set_debug(config, debug_json_path) |
| platform_process.py  | 多平台数据/模型同步与导出       | from openi_utils import platform_preprocess, platform_postprocess |
| openi.py             | OpenI/ModelArts数据与模型同步   | from openi_utils import openi_multidataset_to_env, pretrain_to_env, env_to_openi |
| default_debug.json   | 调试参数json模板                | 作为 set_debug 的参数参考 |

## 使用说明

- 推荐通过 config/config.json 或 argparse 统一管理参数，传递给各主函数。
- set_debug 支持通过 json 文件灵活覆写调试参数，无需修改源代码。
- 详细参数和用法请查阅各脚本头部注释。
- 适合云端/本地混合开发、分布式训练、平台迁移等场景。

## 依赖

- Python 3.6+
- mindspore、moxing、os、json 等（按需安装）

---

如需扩展平台支持或定制自动化流程，请参考各脚本源码进行开发。

