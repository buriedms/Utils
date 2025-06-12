"""
MindSpore分布式与环境设置工具

功能：
- 设置全局随机种子，保证实验可复现。
- 设置设备与并行模式，支持Ascend/GPU/CPU。
- 支持多卡/单卡自动切换。
- 支持AMP混合精度训练等级设置。

依赖：mindspore, numpy, random, os

参数说明：
- config: 需包含 device_target, device_id, device_num, gradients_mean, parameter_broadcast, output_dir, context_mode, amp_level, seed 等字段。

用法：
    from set_environment import set_environment, set_seed, set_device, cast_amp
    set_environment(config)
"""
import mindspore
from mindspore import context, nn
from mindspore.communication import init
import random
import numpy as np
import os
from mindspore import dtype as mstype

def do_keep_fp32(network, cell_types):
    """将指定类型的Cell强制转为float32"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)

def set_seed(config):
    """
    设置全局随机种子，保证实验可复现。
    :param config: 需包含 config.seed
    """
    mindspore.common.set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    mindspore.set_seed(config.seed)
    print(f'\nset global seed : {config.seed}\n')

def set_device(config):
    """
    设置设备与并行模式（支持Ascend/GPU/CPU，多卡/单卡自动切换）
    :param config: 需包含 device_target, device_id, device_num, gradients_mean, parameter_broadcast, output_dir
    """
    device_target = config.device_target
    device_id = config.device_id
    device_num = config.device_num
    assert device_target in ['Ascend', 'GPU', 'CPU']
    if device_target == "Ascend":
        if device_num > 1:
            config.device_id = int(os.environ.get("DEVICE_ID", config.device_id))
            context.set_context(device_id=config.device_id, device_target=config.device_target)
            context.set_context(enable_graph_kernel=False)
            print(f'\nuse multi device: {device_num} local device id: {config.device_id}\n')
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=config.gradients_mean, parameter_broadcast=config.parameter_broadcast)
        else:
            context.set_context(device_id=config.device_id, device_target=config.device_target)
            print(f'\nuse single device local device id: {config.device_id}\n')
    elif device_target == "GPU":
        if device_num > 1:
            config.device_id = int(os.environ.get("DEVICE_ID", config.device_id))
            context.set_context(device_id=config.device_id, device_target=config.device_target)
            context.set_context(enable_graph_kernel=False)
            print(f'\nuse multi device: {device_num} local device id: {config.device_id}\n')
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=config.gradients_mean, parameter_broadcast=config.parameter_broadcast)
        else:
            context.set_context(device_id=config.device_id, device_target=config.device_target)
            print(f'\nuse single device local device id: {config.device_id}\n')
    elif device_target == 'CPU':
        context.set_context(device_target=config.device_target)
        print('\nuse cpu device\n')
    else:
        raise ValueError("Unsupported platform.")
    config.output_dir = os.path.join(config.output_dir, str(config.device_id))

def set_environment(config):
    """
    设置MindSpore训练环境（全局种子、模式、设备等）
    :param config: 需包含 context_mode, 其它见set_seed/set_device
    """
    set_seed(config)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[config.context_mode])
    set_device(config)

def cast_amp(config, net):
    """
    设置网络AMP混合精度等级
    :param config: 需包含 amp_level
    :param net: nn.Cell类型
    """
    if config.amp_level == "O1":
        print(f"=> using amp_level {config.amp_level}\n")
        net.to_float(mstype.float16)
        cell_types = (nn.GELU, nn.ReLU, nn.Softmax, nn.Conv2d, nn.Conv1d, nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif config.amp_level == "O2":
        print(f"=> using amp_level {config.amp_level}\n")
        net.to_float(mstype.float16)
        cell_types = (nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif config.amp_level == "O3":
        print(f"=> using amp_level {config.amp_level}\n")
        net.to_float(mstype.float16)
    elif config.amp_level == 'Ox':
        print(f"=> using amp_level {config.amp_level}\n")
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float16)
    else:
        print(f"=> using amp_level {config.amp_level}")
        config.loss_scale = 1.
        config.is_dynamic_loss_scale = 0
        print(f"=> When amp_level is O0, using fixed loss_scale with {config.loss_scale}")
