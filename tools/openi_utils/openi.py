"""
OpenI/ModelArts 平台数据与模型同步工具

功能：
- 支持单/多数据集自动下载、解压、同步到训练环境。
- 支持预训练模型自动下载。
- 支持训练输出自动上传到云端。
- 支持多卡/分布式环境下的同步与上传。
- 支持通过外部json配置参数，便于调试和自动化。

依赖：os, json, moxing, mindspore

参数说明：
- data_url, data_dir, multi_data_url, pretrain_url, pretrain_dir, train_dir, train_url, keep_name, obs_train_url 等。
- 推荐通过外部json配置（如openi_config.json）进行参数管理。

用法：
    from openi import openi_dataset_to_Env, openi_multidataset_to_env, pretrain_to_env, env_to_openi, obs_copy_file, obs_copy_folder, multidataset_to_env, EnvToOpenIEpochEnd
    # 参考 openi_config.json 进行参数配置
    openi_multidataset_to_env(multi_data_url, data_dir)
    pretrain_to_env(pretrain_url, pretrain_dir)
    env_to_openi(train_dir, train_url)
"""
import os
import json
import moxing as mox
from mindspore.train.callback import Callback

def openi_dataset_to_Env(data_url, data_dir):
    """
    OpenI平台：单数据集下载到训练镜像
    :param data_url: 数据集远程url
    :param data_dir: 本地存储路径
    :return: None
    """
    try:
        mox.file.copy_parallel(data_url, data_dir)
        print(f"Successfully Download {data_url} to {data_dir}")
    except Exception as e:
        print(f'moxing download {data_url} to {data_dir} failed: {e}')
    return

def openi_multidataset_to_env(multi_data_url, data_dir, keep_name=False):
    """
    OpenI平台：多数据集下载到训练镜像
    :param multi_data_url: 多数据集url的json字符串
    :param data_dir: 本地存储路径
    :param keep_name: 是否以zip名为子目录
    :return: None
    """
    multi_data_json = json.loads(multi_data_url)
    for i in range(len(multi_data_json)):
        if keep_name:
            path = os.path.join(data_dir, multi_data_json[i]["dataset_name"])
        else:
            path = data_dir
        if not os.path.exists(path):
            os.makedirs(path)
        try:
            mox.file.copy_parallel(multi_data_json[i]["dataset_url"], path)
            print(f"Successfully Download {multi_data_json[i]['dataset_url']} to {path}")
        except Exception as e:
            print(f'moxing download {multi_data_json[i]["dataset_url"]} to {path} failed: {e}')
    return

def pretrain_to_env(pretrain_url, pretrain_dir):
    """
    OpenI平台：预训练模型下载到训练镜像
    :param pretrain_url: 预训练模型url的json字符串
    :param pretrain_dir: 本地存储路径
    :return: None
    """
    pretrain_url_json = json.loads(pretrain_url)
    print("pretrain_url_json:", pretrain_url_json)
    for i in range(len(pretrain_url_json)):
        modelfile_path = os.path.join(pretrain_dir, pretrain_url_json[i]["model_name"])
        try:
            mox.file.copy(pretrain_url_json[i]["model_url"], modelfile_path)
            print(f"Successfully Download {pretrain_url_json[i]['model_url']} to {modelfile_path}")
        except Exception as e:
            print(f'moxing download {pretrain_url_json[i]["model_url"]} to {modelfile_path} failed: {e}')
    return

def env_to_openi(train_dir, train_url):
    """
    OpenI平台：训练输出自动上传到云端
    :param train_dir: 本地输出目录
    :param train_url: 云端目标url
    :return: None
    """
    device_num = int(os.getenv('RANK_SIZE', '1'))
    local_rank = int(os.getenv('RANK_ID', '0'))
    if device_num == 1:
        obs_copy_folder(train_dir, train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            obs_copy_folder(train_dir, train_url)
    return

def obs_copy_file(obs_file_url, file_url):
    """
    obs与本地/obs间单文件拷贝
    :param obs_file_url: 源文件url
    :param file_url: 目标文件url
    """
    try:
        mox.file.copy(obs_file_url, file_url)
        print(f"Successfully Download {obs_file_url} to {file_url}")
    except Exception as e:
        print(f'moxing download {obs_file_url} to {file_url} failed: {e}')
    return

def obs_copy_folder(folder_dir, obs_folder_url):
    """
    obs与本地/obs间文件夹拷贝
    :param folder_dir: 源文件夹
    :param obs_folder_url: 目标文件夹url
    """
    try:
        mox.file.copy_parallel(folder_dir, obs_folder_url)
        print(f"Successfully Upload {folder_dir} to {obs_folder_url}")
    except Exception as e:
        print(f'moxing upload {folder_dir} to {obs_folder_url} failed: {e}')
    return

def multidataset_to_env(multi_data_url, data_dir, keep_name=False):
    """
    通用平台：多数据集下载并自动解压到训练镜像
    :param multi_data_url: 多数据集url的json字符串
    :param data_dir: 本地存储路径
    :param keep_name: 是否以zip名为子目录
    :return: None
    """
    multi_data_json = json.loads(multi_data_url)
    for i in range(len(multi_data_json)):
        zipfile_path = os.path.join(data_dir, multi_data_json[i]["dataset_name"])
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path)
            print(f"Successfully Download {multi_data_json[i]['dataset_url']} to {zipfile_path}")
            # 解压
            if keep_name:
                filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
                filePath = os.path.join(data_dir, filename)
            else:
                filePath = data_dir
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            os.system(f"unzip {zipfile_path} -d {filePath}")
        except Exception as e:
            print(f'moxing download {multi_data_json[i]["dataset_url"]} to {zipfile_path} failed: {e}')
    return

class EnvToOpenIEpochEnd(Callback):
    """
    训练每个epoch结束时自动上传输出到openi
    """
    def __init__(self, train_dir, obs_train_url):
        self.train_dir = train_dir
        self.obs_train_url = obs_train_url
    def epoch_end(self, run_context):
        obs_copy_folder(self.train_dir, self.obs_train_url)
