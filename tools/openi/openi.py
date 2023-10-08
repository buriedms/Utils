import os
import json
import moxing as mox
from mindspore.train.callback import Callback


def openi_dataset_to_Env(data_url, data_dir):
    """
    启智集群：openi copy single dataset to training image

    :param data_url:数据的远程路径url
    :param data_dir:数据在训练环境当中需要的目标存储位置
    :return:None
    """
    try:
        mox.file.copy_parallel(data_url, data_dir)
        print("Successfully Download {} to {}".format(data_url, data_dir))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(data_url, data_dir) + str(e))
    return


def openi_multidataset_to_env(multi_data_url, data_dir, keep_name=False):
    """
    启智集群：copy single or multi dataset to training image
    :param multi_data_url:官方超参数，多个数据集的url路径
    :param data_dir:数据在训练环境当中需要的目标存储位置
    :param keep_name: 是否保留zip文件名字作为上一级目录
    :return: None
    """
    multi_data_json = json.loads(multi_data_url)
    for i in range(len(multi_data_json)):
        if keep_name:
            path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        else:
            path = data_dir
        if not os.path.exists(path):
            os.makedirs(path)
        """case 1"""
        try:
            mox.file.copy_parallel(multi_data_json[i]["dataset_url"], path)
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"], path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], path) + str(e))
        """case 2"""
        # openi_dataset_to_Env(multi_data_json[i]["dataset_url"], path)
    return


def pretrain_to_env(pretrain_url, pretrain_dir):
    """
    均可使用：copy pretrain to training image
    :param pretrain_url: 预训练模型的远程路径url
    :param pretrain_dir: 预训练模型训练环境当中保存位置
    :return:
    """
    pretrain_url_json = json.loads(pretrain_url)
    print("pretrain_url_json:", pretrain_url_json)
    for i in range(len(pretrain_url_json)):
        modelfile_path = pretrain_dir + "/" + pretrain_url_json[i]["model_name"]
        try:
            mox.file.copy(pretrain_url_json[i]["model_url"], modelfile_path)
            print("Successfully Download {} to {}".format(pretrain_url_json[i]["model_url"], modelfile_path))
        except Exception as e:
            print('moxing download {} to {} failed: '.format(pretrain_url_json[i]["model_url"], modelfile_path) + str(e))
    return


def env_to_openi(train_dir, train_url):
    """
    均可使用：upload output to openi
    :param train_dir: 训练结果文件在训练环境当中的存储位置
    :param train_url: 训练结果文件上传的远程路径url
    :return: None
    """
    device_num = int(os.getenv('RANK_SIZE'))
    local_rank = int(os.getenv('RANK_ID'))
    if device_num == 1:
        obs_copy_folder(train_dir, train_url)
    if device_num > 1:
        if local_rank % 8 == 0:
            obs_copy_folder(train_dir, train_url)
    return


def obs_copy_file(obs_file_url, file_url):
    """
    cope file from obs to obs, or cope file from obs to env, or cope file from env to obs
    """
    try:
        mox.file.copy(obs_file_url, file_url)
        print("Successfully Download {} to {}".format(obs_file_url, file_url))
    except Exception as e:
        print('moxing download {} to {} failed: '.format(obs_file_url, file_url) + str(e))
    return


def obs_copy_folder(folder_dir, obs_folder_url):
    """
    copy folder from obs to obs, or copy folder from obs to env, or copy folder from env to obs
    """
    try:
        mox.file.copy_parallel(folder_dir, obs_folder_url)
        print("Successfully Upload {} to {}".format(folder_dir, obs_folder_url))
    except Exception as e:
        print('moxing upload {} to {} failed: '.format(folder_dir, obs_folder_url) + str(e))
    return


def multidataset_to_env(multi_data_url, data_dir, keep_name=False):
    """
    智算平台：copy single or multi dataset to training image
    :param multi_data_url:官方参数，多个数据集的url路径
    :param data_dir:数据在训练环境当中需要的目标存储位置
    :param keep_name: 是否保留zip文件名字作为上一级目录
    :return: None
    """
    multi_data_json = json.loads(multi_data_url)
    for i in range(len(multi_data_json)):
        zipfile_path = data_dir + "/" + multi_data_json[i]["dataset_name"]
        try:
            mox.file.copy(multi_data_json[i]["dataset_url"], zipfile_path)
            print("Successfully Download {} to {}".format(multi_data_json[i]["dataset_url"], zipfile_path))
            # get filename and unzip the dataset
            if keep_name:
                filename = os.path.splitext(multi_data_json[i]["dataset_name"])[0]
                filePath = data_dir + "/" + filename
            else:
                filePath = data_dir
            if not os.path.exists(filePath):
                os.makedirs(filePath)
            # If it is a tar compressed package, you can use os.system("tar -xvf {} {}".format(zipfile_path, filePath))
            os.system("unzip {} -d {}".format(zipfile_path, filePath))

        except Exception as e:
            print('moxing download {} to {} failed: '.format(
                multi_data_json[i]["dataset_url"], zipfile_path) + str(e))
    return


class EnvToOpenIEpochEnd(Callback):
    """
    upload output to openi when epoch end
    """

    def __init__(self, train_dir, obs_train_url):
        self.train_dir = train_dir
        self.obs_train_url = obs_train_url

    def epoch_end(self, run_context):
        obs_copy_folder(self.train_dir, self.obs_train_url)
