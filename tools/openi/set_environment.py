import mindspore
from mindspore import context, nn
from mindspore.communication import init, get_rank, get_group_size
import random
import numpy as np
import os
from mindspore import dtype as mstype


def do_keep_fp32(network, cell_types):
    """Cast cell to fp32 if cell in cell_types"""
    for _, cell in network.cells_and_names():
        if isinstance(cell, cell_types):
            cell.to_float(mstype.float32)


def set_seed(config):
    """
    设置全局使用的种子，保证训练的可复现。
    :param config: 参数类型，需要包含参数config.seed
    :return: None
    """
    mindspore.common.set_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    mindspore.set_seed(config.seed)
    print('\nset global seed : {}\n'.format(config.seed))


def set_device(args):
    """
    Set device and ParallelMode(if device_num > 1)
    根据device_target设置训练环境的设备类型
    根据device_id&device_num设置训练所使用的卡数目和对应的卡id
    如果为多卡模式，则会更根据gradients_mean和parameter_broadcast初始化设置梯度的平均行为，和初始化将0卡参数广播到其余的卡上
    如果为多卡模式，在输出路径output_dir当中加入子目录device_id

    :param args: 参数类型，需要包含参数[args.device_target, args.device_id, args.device_num, args.gradients_mean, args.parameter_broadcast, args.output_dir]
    :return: None
    """
    device_target = args.device_target
    device_id = args.device_id
    device_num = args.device_num

    assert device_target in ['Ascend', 'GPU', 'CPU']

    if device_target == "Ascend":
        if device_num > 1:
            args.device_id = int(os.environ.get("DEVICE_ID", args.device_id))
            # context.set_context(device_id=device_id)
            context.set_context(device_id=args.device_id, device_target=args.device_target)
            context.set_context(enable_graph_kernel=False)
            print('\nuse multi device: {} local device id: {}\n'.format(device_num, args.device_id))

            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=args.gradients_mean, parameter_broadcast=args.parameter_broadcast)

            # context.set_auto_parallel_context(pipeline_stages=2, full_batch=True)
            # args.device_id = get_rank()
        else:
            context.set_context(device_id=args.device_id, device_target=args.device_target)
            print('\nuse single device local device id: {}\n'.format(args.device_id))
    elif device_target == "GPU":
        if device_num > 1:
            args.device_id = int(os.environ.get("DEVICE_ID", args.device_id))
            context.set_context(device_id=args.device_id, device_target=args.device_target)
            context.set_context(enable_graph_kernel=False)
            print('\nuse multi device: {} local device id: {}\n'.format(device_num, args.device_id))
            # context.set_context(device_id=device_id)
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              gradients_mean=args.gradients_mean, parameter_broadcast=args.parameter_broadcast)

            # args.device_id = get_rank()

        else:
            context.set_context(device_id=args.device_id, device_target=args.device_target)
            print('\nuse single device local device id: {}\n'.format(args.device_id))
    elif device_target == 'CPU':
        context.set_context(device_target=args.device_target)
        print('\nuse cpu device\n')
    else:
        raise ValueError("Unsupported platform.")
    args.output_dir = os.path.join(args.output_dir, str(args.device_id))


def set_environment(config):
    """
    设置训练环境，包括全局种子、训练模式、训练设备设置
    :param config: 参数类型
    :return: None

    """
    set_seed(config)
    mode = {
        0: context.GRAPH_MODE,
        1: context.PYNATIVE_MODE
    }
    context.set_context(mode=mode[config.context_mode])
    set_device(config)


def cast_amp(args, net):
    """
    cast network amp_level
    :param args: 参数类型，需要包含参数args.amp_level
    :param net: nn.Cell类型，对模型进行转换精度类型
    :return: None
    """
    if args.amp_level == "O1":
        print(f"=> using amp_level {args.amp_level}\n")
        net.to_float(mstype.float16)
        cell_types = (nn.GELU, nn.ReLU, nn.Softmax, nn.Conv2d, nn.Conv1d, nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O2":
        print(f"=> using amp_level {args.amp_level}\n")
        net.to_float(mstype.float16)
        cell_types = (nn.BatchNorm2d, nn.LayerNorm)
        print(f"=> cast {cell_types} to fp32 back")
        do_keep_fp32(net, cell_types)
    elif args.amp_level == "O3":
        print(f"=> using amp_level {args.amp_level}\n")
        net.to_float(mstype.float16)
    elif args.amp_level == 'Ox':
        print(f"=> using amp_level {args.amp_level}\n")
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.to_float(mstype.float16)
    else:
        print(f"=> using amp_level {args.amp_level}")
        args.loss_scale = 1.
        args.is_dynamic_loss_scale = 0
        print(f"=> When amp_level is O0, using fixed loss_scale with {args.loss_scale}")
