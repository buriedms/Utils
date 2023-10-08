# logger的撰写范本
"""
生成指定的logger范本
"""
import os
import logging


def gen_logger(save_path=None, name=None, chlr=False, mode='w'):
    """
    logger的范本，可以设定保存路径和输出到控制台
    :param save_path:logger保存的路径，default: '.'
    :param name:logger保存的名字，default: 'log.log'
    :param chlr:是否输出到控制台，default: False
    :param mode: 输出到文件的模式, mode=['w','a'], default: 'w'
    :return logger
    """
    name = 'log.log' if not name else name
    save_path = '.' if not save_path else save_path
    file_path = os.path.join(save_path, name)
    print(file_path)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=file_path,
        filemode=mode
    )
    logger = logging.getLogger()
    if chlr and file_path:
        chlr = logging.StreamHandler()  # 输出到控制台的handler
        logger.addHandler(chlr)
    return logger


if __name__ == '__main__':
    # logger=gen_logger(save_path='text.log')
    # logger.info('你好!!!')
    # logger.info('hello')
    os.makedirs('AdaptSegNet/data/GTA5', exist_ok=True)
    os.makedirs('AdaptSegNet/data/Cityscapes', exist_ok=True)
