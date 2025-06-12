# cli_tools

命令行辅助工具包

## 功能简介

本目录下包含一系列可直接通过命令行调用的数据处理与文件操作工具，适用于批量图片处理、文件复制、解压等常见任务。

## 工具列表

| 脚本名                | 功能描述                 | 典型用法示例 |
|----------------------|------------------------|-------------|
| image_concat.py      | 批量拼接两个文件夹下的图片 | python image_concat.py --src1 dir1 --src2 dir2 --dst out |
| unzips.py            | 批量解压指定目录下的zip文件 | python unzips.py --src zip_dir --dst out_dir |
| copy_by_txt.py       | 按txt文件内容批量复制文件   | python copy_by_txt.py --txt file.txt --dst out_dir |
| multi_process.py     | 多进程运行模板             | 作为范本参考 |
| read_excel_write_txt.py | excel内容提取到txt      | python read_excel_write_txt.py --excel file.xlsx --txt out.txt |

## 使用说明

- 每个脚本均可通过 `python 脚本名.py --help` 查看详细参数说明。
- 建议将本目录加入环境变量或在tools目录下直接运行。
- 适合日常数据预处理、文件批量操作等场景。

## 依赖

- Python 3.6+
- 可能依赖 openpyxl、Pillow 等第三方库，详见各脚本头部说明。

---

如需定制功能或有批量处理需求，可参考各脚本源码进行扩展。

