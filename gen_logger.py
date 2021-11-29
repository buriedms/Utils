import os

def gen_logger(save_path=None,name=None,chlr=False,mode='w'):
    import logging
    import os
    name='' if not name else name
    save_path='' if not save_path else save_path
    file_path=os.path.join(save_path,name)
    print(file_path)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO ,
        filename=file_path,
        filemode=mode
    )
    logger=logging.getLogger()
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
