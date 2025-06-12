"""
MindSpore ckpt参数查看工具

功能：
- 查看ckpt文件的参数名和参数内容。
- 支持输出到log文件或直接返回。
- 适用于模型参数结构检查、调试、迁移等。

依赖：mindspore, os
"""
import os
import mindspore
from typing import List, Tuple, Optional

def ckpt_view(
    ckpt_path: str,
    log_path: Optional[str] = None,
    to_log: bool = True
) -> List[Tuple[str, mindspore.Tensor]]:
    """
    查看ckpt文件参数名和内容，并可输出到log文件

    :param ckpt_path: ckpt文件路径
    :param log_path: log文件路径（默认与ckpt同名）
    :param to_log: 是否输出到log文件
    :return: (参数名, 参数内容)元组列表
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'ckpt路径不存在: {ckpt_path}')
    if to_log and not log_path:
        log_path = ckpt_path.replace('.ckpt', '_param_state.log')
    state_dict = mindspore.load_checkpoint(ckpt_path)
    result = [(name, state_dict[name]) for name in sorted(state_dict.keys())]
    if to_log:
        with open(log_path, 'w', encoding='utf-8') as logger:
            logger.write(f'参数总数: {len(result)}\n')
            for name, param in result:
                logger.write(f'{name}: {param}\n')
        print(f'参数信息已输出到: {log_path}')
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='MindSpore ckpt参数查看工具')
    parser.add_argument('--ckpt', required=True, help='ckpt文件路径')
    parser.add_argument('--nolog', action='store_true', help='不输出到log文件')
    parser.add_argument('--log', default=None, help='log文件路径（可选）')
    args = parser.parse_args()
    try:
        result = ckpt_view(
            ckpt_path=args.ckpt,
            log_path=args.log,
            to_log=not args.nolog
        )
        print(f'参数数量: {len(result)}')
        print('前5个参数:', result[:5])
    except Exception as e:
        print(f'执行失败: {e}')
