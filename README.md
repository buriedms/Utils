# Utils

我的常用工具包

## 目录

<!-- TOC -->

* [1. image_concat.py](#1-imageconcatpy)
* [2. unzips.py](#2-unzipspy)
* [3. ckpt_view.py](#3-ckptviewpy)
* [4. copy_by_txt.py](#4-copybytxtpy)
* [5. gen_logger.py](#5-genloggerpy)
* [6. multi_process.py](#6-multiprocesspy)
* [7. read_excel_write_txt.py](#7-readexcelwritetxtpy)
* [8. pdparams2ckpt.py](#8-pdparams2ckptpy)
* [9. pth2ckpt.py](#9-pth2ckptpy)
* [10. pth2pdparams.py](#10-pth2pdparamspy)
* [11. openi.py](#11-openipy)
* [12. platform_process.py](#12-platformprocesspy)
* [13. set_debug.py](#13-setdebugpy)
* [14. set_environment.py](#14-setenvironmentpy)

<!-- TOC -->

## 工具内容

### [1. image_concat.py](tools/image_concat.py)

**功能**：拼接两个文件夹当中的图像

**原理**：None

**支持平台**：paddle

**参数解析**

| 参 数          | 说明               | 
|--------------|------------------|
| input_path_1 | 合并的图片的第一个文件夹路径   | 
| input_path_2 | 合并的图片的第二个文件夹路径   | 
| output_path  | 输出合并图片的路径        | 
| save_name    | 保存图片的名称，默认result | 

**案例效果**
![图1](imgs/result.png)

### [2. unzips.py](tools/unzips.py)

**功能**：批量解压同一目录下多个zip文件

**原理**：调用本地7zip程序进行解压

**参数解析**

| 参数             | 说明           |
|----------------|--------------|
| ZIP_LIST       | 指定可访问的压缩包形式  |
| EXT_LIST       | 指定可识别的压缩包拓展名 |
| ZIP7_PATH      | 解压程序所在路径     |
| FOLD_PATH_FROM | 待解压的压缩包源路径   |
| FOLD_PATH_TO   | 所有文件解压的解压位置  |

**支持平台**：All

### 3. [ckpt_view.py](tools/ckpt_view.py)

**功能**：查看指定ckpt文件的参数名和内置参数情况，默认可以输出到相同位置的log文件，也同样具有返回值return

**原理**：None

**参数解析**

| 参数            | 说明                                                         |
|---------------|------------------------------------------------------------|
| ckpt_path     | 需要查看的ckpt路径                                                |
| return_name   | 返回值是否包含name，两者均为false则返回name                               |
| return_param  | 返回值是否包含param，两者均为false则返回name                              |
| return_logger | 是否将结果定向输出到log当中                                            |
| logger_path   | 结果输出的logger路径，默认和ckpt_path相同                               |
| return        | 根据关键字指定返回值类型 [Optional: (names,params), (names), (params)] |

**支持平台**：mindspore

### [4. copy_by_txt.py](tools/copy_by_txt.py)

**功能**：根据txt内容，从指定目录下复制文件到目标路径

**原理**：使用shutil.copy实现

**参数解析**

| 参数             | 说明                                       |
|----------------|------------------------------------------|
| fold_path_from | 源文件目录所在路径                                |
| fold_path_to   | 目标目录所在路径                                 |
| txt_path       | 选定的copy文件内容的txt文件所在路径                    |
| extend         | 当txt内容仅有文件名，没有后缀时，可以设定该项，例如extend='.png' |
| return         | None                                     |

**支持平台**：All

### [5. gen_logger.py](./tools/gen_logger.py)

**功能**：logger的范本，可以设定保存路径和输出到控制台

**原理**：调用logging库进行撰写

**参数解析**

| 参数        | 说明                                     |
|-----------|----------------------------------------|
| save_path | logger保存的路径，default: '.'               |
| name      | logger保存的名字，default: 'log.log'         |
| chlr      | 是否输出到控制台，default: False                |
| mode      | 输出到文件的模式, mode=['w','a'], default: 'w' |
| return    | logger                                 |

**支持平台**：All

### [6. multi_process.py](tools/multi_process.py)

**功能**：多进程运行的撰写范本

**原理**：使用threading库进行实现

**参数解析**

None

**支持平台**：All

### [7. read_excel_write_txt.py](./tools/read_excel_write_txt.py)

**功能**：阅读excel当中内容，并且提取指定内容到txt当中

**原理**：通过openpyxl库和numpy库进行实现

**参数解析**

| 函数         | 参数        | 说明                  |
|------------|-----------|---------------------|
| read_excel | path      | xlsx文件路径            |
| -          | rows      | 读取的表格行范围,如果为负数,则从后记 |
| -          | cols      | 读取的表格列范围,如果为负数,则从后记 |
| -          | data_only | 仅读取其中数据             |
| -          | return    | excel表格当中内容         |
| case_1     | path      | xlsx文件路径            |
| -          | rows      | 读取的excel表格范围        |
| -          | cols      | 读取的excel表格范围        |
| -          | return    | 保存在当前路径下,格式为txt     |

**支持平台**：All

### [8. pdparams2ckpt.py](./tools/model_transform/pdparams2ckpt.py)

**功能**：转换paddle的模型参数为torch的模型参数

**原理**：更改参数名, 保持数据不变, 更改参数的实例类别

**参数解析**

None, 仅提供案例

**支持平台**：paddle & torch

### [9. pth2ckpt.py](./tools/model_transform/pth2ckpt.py)

**功能**：转换torch的模型参数为mindspore的模型参数

**原理**：更改参数名, 保持数据不变, 更改参数的实例类别

**参数解析**

| 函数                 | 参数             | 说明                                                                             |
|--------------------|----------------|--------------------------------------------------------------------------------|
| update_torch_to_ms | pth_saved_path | pth load path                                                                  |
| -                  | ms_save_path   | ckpt save path                                                                 |
| -                  | txt_save_path  | pth 2 ckpt log save path                                                       |
| -                  | space          | space: control the space in log file, default: 80                              |
| -                  | name_change    | the parameter's name change rule: pth to ckpt ; type: dict{pth_name:ckpt_name} |
| -                  | filter_list    | The filter list of parameters : list ['pth_name1','pth_name2']                 |
| -                  | return         | the changed ckpt                                                               |
| case_1             | -              | -                                                                              |
| case_2             | -              | -                                                                              |

**支持平台**：torch & mindspore

### [10. pth2pdparams.py](./tools/model_transform/pth2pdparams.py)

**功能**：转换torch的模型参数为mindspore的模型参数

**原理**：更改参数名, 保持数据不变, 更改参数的实例类别

**参数解析**

None, 仅提供案例

**支持平台**：paddle & torch

### [11. openi.py](./tools/openi/openi.py)

**功能**：实现数据文件从obs桶和运行环境之间的相互转移

**原理**：通过官方moxing库进行实现，并且对于特殊zip文件，使用unzip命令进行解压

**参数解析**

| 函数                        | 参数             | 说明                                                  |
|---------------------------|----------------|-----------------------------------------------------|
| openi_dataset_to_Env      | -              | 启智集群：openi copy single dataset to training image    |
| -                         | data_url       | 数据的远程路径url                                          |
| -                         | data_dir       | 数据在训练环境当中需要的目标存储位置                                  |
| openi_multidataset_to_env | -              | 启智集群：copy single or multi dataset to training image |
| -                         | multi_data_url | 官方参数，多个数据集的url路径                                    |
| -                         | data_dir       | 数据在训练环境当中需要的目标存储位置                                  |
| -                         | keep_name      | 是否保留zip文件名字作为上一级目录                                  |
| multidataset_to_env       | -              | 智算平台：copy single or multi dataset to training image |
| -                         | multi_data_url | 官方参数，多个数据集的url路径                                    |
| -                         | data_dir       | 数据在训练环境当中需要的目标存储位置                                  |
| -                         | keep_name      | 是否保留zip文件名字作为上一级目录                                  |
| pretrain_to_env           | -              | 均可使用：copy pretrain to training image                |
| -                         | pretrain_url   | 预训练模型的远程路径url                                       |
| -                         | pretrain_dir   | 预训练模型训练环境当中保存位置                                     |
| env_to_openi              | -              | 均可使用：upload output to openi                         |
| -                         | train_dir      | 训练结果文件在训练环境当中的存储位置                                  |
| -                         | train_url      | 训练结果文件上传的远程路径url                                    |

**支持平台 **：mindspore

### [12. platform_process.py](./tools/openi/platform_process.py)

**功能**：实现多个平台的数据载入和导出过程。预设：启智集群，智算集群，modelarts。测试通过：启智集群，智算集群

**原理**：调用[openi.py](./tools/openi/openi.py)文件进行实现，  
对于训练数据统一保存到当前训练环境的/cache/data目录，  
对于模型文件统一保存到当前训练环境的/cache/pretrained目录，  
对于输出目录统一设置为/cache/output目录。  
通过软连接命令`ln -s [source path] [target path]`将文件目录进行连接起来，具体代码：

```shell
ln -s /cache/data ./data
ln -s /cache/pretrained ./pretrained
ln -s /cache/output ./output
```

因此具体使用时，从当前运行环境的`./data`目录进行训练数据读取，
从`./pretrained`进行预训练模型读取，
将训练日志及相应模型信息保存到`./output`即可

**参数解析**

| 函数                  | 参数              | 说明                       |
|---------------------|-----------------|--------------------------|
| platform_preprocess | args            | 参数类型,需要具备以下参数            |
| -                   | args.model_arts | 当前平台是否为modelarts         |
| -                   | args.openi      | 当前平台是否为openi             |
| -                   | args.platform   | 若平台为openi,则使用的集群是否为ZS或QZ |
| -                   | args.device_id  | 当前训练环境所使用的卡id            |
| -                   | args.device_num | 当前训练环境所使用的卡数目            |

### [13. set_debug.py](./tools/openi/set_debug.py)

**功能**：在debug超参数为True时，修改某些参数的值，减小消耗显存使之跑通。

**原理**：None

**参数解析**

| 参数           | 说明           |
|--------------|--------------|
| config       | 参数类型，以下为必要参数 |
| config.debug | 是否启用debug    |

注意： 具体内容根据实际情况进行自定义设置

### [14. set_environment.py](./tools/openi/set_environment.py)

**功能**：根据设定的超参数，初始化训练环境，包括种子设置、多卡相关设置等等，另外包含混合精度相关设置函数

**原理**：通过mindspore官方库相关函数进行实现

**参数解析**：

| 函数              | 参数     | 说明                                                                                                                               |
|-----------------|--------|----------------------------------------------------------------------------------------------------------------------------------|
| set_seed        | -      | 设置全局使用的种子，保证训练的可复现。                                                                                                              |
| -               | config | 参数类型，需要包含参数config.seed                                                                                                           |
| set_device      | -      | Set device and ParallelMode(if device_num > 1)                                                                                   |
| -               | args   | 参数类型，需要包含参数[args.device_target, args.device_id, args.device_num, args.gradients_mean, args.parameter_broadcast, args.output_dir] |
| set_environment | -      | 设置训练环境，包括全局种子、训练模式、训练设备设置                                                                                                        |
| -               | config | 参数类型                                                                                                                             |
| cast_amp        | -      | cast network amp_level                                                                                                           |
| -               | args   | 参数类型，需要包含参数args.amp_level                                                                                                        |
| -               | net    | nn.Cell类型，对模型进行转换精度类型                                                                                                            |

**支持平台**: Mindspore