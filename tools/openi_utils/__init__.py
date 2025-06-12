"""
openi_utils

OpenI/ModelArts 平台环境、数据、调试等自动化工具包。

常用导出：
- set_environment, set_seed, set_device, cast_amp
- set_debug
- platform_preprocess, platform_postprocess
- openi_dataset_to_Env, openi_multidataset_to_env, pretrain_to_env, env_to_openi, obs_copy_file, obs_copy_folder, multidataset_to_env, EnvToOpenIEpochEnd
"""
from .set_environment import set_environment, set_seed, set_device, cast_amp
from .set_debug import set_debug
from .platform_process import platform_preprocess, platform_postprocess
from .openi import (
    openi_dataset_to_Env, openi_multidataset_to_env, pretrain_to_env, env_to_openi,
    obs_copy_file, obs_copy_folder, multidataset_to_env, EnvToOpenIEpochEnd
)

