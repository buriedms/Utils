"""
查看指定ckpt文件的参数名和内置参数情况，默认可以输出到相同位置的log文件，也同样具有返回值return
"""

import os
import mindspore


def ckpt_view(ckpt_path: str, return_name=False, return_param=False, return_logger=True, logger_path: str = None):
    """
    查看指定ckpt文件的参数名和内置参数情况，默认可以输出到相同位置的log文件，也同样具有返回值return
    :param ckpt_path: 需要查看的ckpt路径
    :param return_name: 返回值是否包含name，两者均为false则返回name
    :param return_param: 返回值是否包含param，两者均为false则返回name
    :param return_logger: 是否将结果定向输出到log当中
    :param logger_path: 结果输出的logger路径，默认和ckpt_path相同
    :return:根据关键字指定返回值类型
    [Optional: (names,params), (names), (params)]
    """
    if not os.path.exists(ckpt_path):
        raise ValueError('The ckpt path is not exist : {} , please check it.')
    if not logger_path and return_logger:
        logger_path = ckpt_path.replace('.ckpt', '_param_state.log')
    state_dict = mindspore.load_checkpoint(ckpt_path)

    names = sorted(list(state_dict.keys()))
    params = list(state_dict[key] for key in names)

    if return_name and return_param:
        return_item = zip(names, params)
    elif return_param:
        return_item = zip(params)
    else:
        return_item = zip(names)

    if return_logger:
        logger = open(logger_path, 'w')
        logger.write('The total number of parameters:{}\n'.format(len(names)))
        for i, item in enumerate(return_item):
            # print('{}:{}'.format(i+1, item))
            # logger.write('{}:{:}\n'.format(i + 1, item))
            logger.write('{:}\n'.format( item))
        print('The log output path:\t{}'.format(logger_path))
        logger.close()

    return return_item


if __name__ == '__main__':
    # ckpt_path = r"D:\Files\GitHub\Utils\temp\checkpoints\swin_transformer_384_version2.ckpt"
    ckpt_path = r"D:\Files\GitHub\Utils\temp\checkpoints\hrnetv2_w18_imagenet_pretrained.ckpt"

    # ckpt_path =r"D:\Files\GitHub\Utils\temp\DeepLab_resnet_pretrained_imagenet.ckpt"
    # ckpt_path = r"D:\Files\GitHub\Utils\temp\Pretrain_DeeplabMulti.ckpt"
    ckpt_view(ckpt_path=ckpt_path, return_param=True, return_name=True)
    # ckpt_view(ckpt_path=ckpt_path, return_name=True)
