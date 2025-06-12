"""
日志logger生成工具

功能：
- 快速生成带文件和控制台输出的logger。
- 支持自定义保存路径、文件名、输出模式。

用法示例：
    logger = gen_logger(save_path='.', name='log.log', chlr=True, mode='w')
    logger.info('日志内容')

依赖：os, logging
"""
import os
import logging

def gen_logger(save_path=None, name=None, chlr=False, mode='w'):
    """
    生成logger，可设定保存路径和输出到控制台
    :param save_path: 日志保存路径，默认当前目录
    :param name: 日志文件名，默认log.log
    :param chlr: 是否输出到控制台
    :param mode: 文件写入模式，'w'覆盖，'a'追加
    :return: logger对象
    """
    name = 'log.log' if not name else name
    save_path = '.' if not save_path else save_path
    file_path = os.path.join(save_path, name)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=file_path,
        filemode=mode
    )
    logger = logging.getLogger()
    if chlr and file_path:
        chlr_handler = logging.StreamHandler()
        logger.addHandler(chlr_handler)
    return logger

if __name__ == '__main__':
    logger = gen_logger(save_path='.', name='text.log', chlr=True)
    logger.info('你好!!!')
    logger.info('hello')
    os.makedirs('AdaptSegNet/data/GTA5', exist_ok=True)
    os.makedirs('AdaptSegNet/data/Cityscapes', exist_ok=True)
