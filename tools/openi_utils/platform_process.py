"""
平台数据与环境预处理/后处理工具

功能：
- 支持ModelArts与OpenI平台的数据、模型、输出目录自动链接与同步。
- 支持多平台数据集、预训练模型、输出结果的自动拷贝与上传。
- 支持平台类型自动识别与断言。

依赖：os, time, subprocess, moxing

参数说明：
- config: 需包含 model_arts, openi, platform, device_id, device_num, data_url, ckpt_url, train_url, multi_data_url, pretrain_url, model_url, output_dir 等字段。

用法：
    from platform_process import platform_preprocess, platform_postprocess
    platform_preprocess(config)
    # ...训练过程...
    platform_postprocess(config)
"""
import subprocess
import os
import time

def cmd_shell(cmd, shell=True):
    print('exec {}'.format(cmd))
    p = subprocess.Popen(cmd, shell=shell)
    p.wait()

def platform_preprocess(config):
    """
    平台数据与环境预处理
    :param config: 平台相关参数
    :return: True/False
    """
    assert not config.model_arts or not config.openi, 'Only one platform can be selected'
    if not config.model_arts and not config.openi:
        print('Not use any platform')
        return False
    if config.device_id % 8 != 0 and config.device_num > 1:
        while not os.path.exists("/cache/download_input.txt"):
            time.sleep(1)
    platform = None
    platform = 'ModelArts' if config.model_arts else platform
    platform = 'Openi-{}'.format(config.platform) if config.openi else platform
    local_data_dir = '/cache/data'
    local_ckpt_dir = '/cache/pretrained'
    local_output_dir = '/cache/output'
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(local_ckpt_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)
    cmd_shell("ln -s {} {}".format(local_data_dir, './data'), shell=True)
    cmd_shell('ln -s {} {}'.format(local_ckpt_dir, './pretrained'), shell=True)
    cmd_shell('ln -s {} {}'.format(local_output_dir, './output'), shell=True)
    config.output_dir = './output'
    if config.model_arts:
        import moxing
        remote_data_url = config.data_url
        remote_ckpt_url = config.ckpt_url
        moxing.file.copy_parallel(src_url=remote_data_url, dst_url=local_data_dir)
        moxing.file.copy_parallel(src_url=remote_ckpt_url, dst_url=local_ckpt_dir)
    if config.openi:
        import moxing
        assert config.platform in ["QZ", "ZS"]
        if config.platform == 'ZS':
            from .openi import c2net_multidataset_to_env as DatasetToEnv
            from .openi import pretrain_to_env
            DatasetToEnv(config.multi_data_url, local_data_dir)
            pretrain_to_env(config.pretrain_url, local_ckpt_dir)
        if config.platform == 'QZ':
            from .openi import openi_multidataset_to_env as DatasetToEnv
            from .openi import pretrain_to_env
            DatasetToEnv(config.multi_data_url, local_data_dir)
            pretrain_to_env(config.pretrain_url, local_ckpt_dir)
    f = open("/cache/download_input.txt", 'w')
    f.close()
    try:
        if os.path.exists("/cache/download_input.txt"):
            print("download_input succeed")
    except Exception as e:
        print("download_input failed")
    print('Use platform : {} , preprocess Over!'.format(platform))
    return True

def platform_postprocess(config):
    """
    平台数据与环境后处理
    :param config: 平台相关参数
    :return: True/False
    """
    assert not config.model_arts or not config.openi, 'Only one platform can be selected'
    if not config.model_arts and not config.openi:
        print('Not use any platform')
        return False
    platform = None
    platform = 'ModelArts' if config.model_arts else platform
    platform = 'Openi-{}'.format(config.platform) if config.openi else platform
    local_output_dir = './output'
    if config.model_arts:
        import moxing
        moxing.file.copy_parallel(src_url=local_output_dir, dst_url=config.train_url)
    if config.openi:
        import moxing
        assert config.platform in ["QZ", "ZS"]
        if config.platform == 'ZS':
            from .openi import env_to_openi
            env_to_openi(local_output_dir, config.model_url)
        if config.platform == 'QZ':
            from .openi import env_to_openi
            env_to_openi(local_output_dir, config.train_url)
    print('Use platform : {} , postprocess Over!'.format(platform))
    return True
