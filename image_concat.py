import paddle
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import pathlib
# import torch
import math
import warnings
import numpy as np
import paddle.vision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont, ImageColor
import os


def data_transforms(img, method=Image.BILINEAR, size=None):
    """
    :param img: 待转换的图片，Image格式
    :param method:插值的方法
    :param size: 转换的大小
    :return: 转换后的图片，Image格式
    """
    return img.resize((300,400), method)


@paddle.no_grad()
def make_grid(
        tensor: Union[paddle.Tensor, List[paddle.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[Tuple[int, int]] = None,
        scale_each: bool = False,
        pad_value: int = 0,
        **kwargs
) -> paddle.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (paddle.is_tensor(tensor) or
            (isinstance(tensor, list) and all(paddle.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if "range" in kwargs.keys():
        warning = "range will be deprecated, please use value_range instead."
        warnings.warn(warning)
        value_range = kwargs["range"]

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = paddle.stack(tensor, axis=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = paddle.concat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = paddle.concat((tensor, tensor, tensor), 1)

    if normalize is True:
        if value_range is not None:
            assert isinstance(value_range, tuple), \
                "value_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, low, high):
            img.clip(min=low, max=high)
            img = img / max(high - low, 1e-5)

        def norm_range(t, value_range):
            if value_range is not None:
                norm_ip(t, value_range[0], value_range[1])
            else:
                norm_ip(t, float(paddle.min(t)), float(paddle.max(t)))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, value_range)
        else:
            norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] +
                                                        padding)
    num_channels = tensor.shape[1]
    grid = paddle.full((num_channels, height * ymaps + padding,
                        width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, x * width + padding:(x + 1) * width] = tensor[k]
            k = k + 1
    return grid


@paddle.no_grad()
def save_image(
        tensor: Union[paddle.Tensor, List[paddle.Tensor]],
        fp: Union[Text, pathlib.Path, BinaryIO],
        format: Optional[str] = None,
        **kwargs
) -> None:
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        **kwargs: Other arguments are documented in ``make_grid``.
    """

    grid = make_grid(tensor, **kwargs)
    ndarr = paddle.clip(grid * 255 + 0.5, 0, 255).transpose(
        [1, 2, 0]).cast("uint8").numpy()
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


def image_dataloader(path, data_transfrom=None):
    print(path)
    input_loader = os.listdir(path)
    input_loader.sort()
    img_transform = transforms.Compose(
        [transforms.Normalize(mean=(127.5, 127.5, 127.5), std=(127.5, 127.5, 127.5), data_format='HWC'),
         transforms.ToTensor()]
    )
    print('data loading')
    inputs = None
    for i in range(len(input_loader)):
        input_name = input_loader[i]
        input_file = os.path.join(path, input_name)
        input = Image.open(input_file).convert("RGB")
        if data_transfrom != None:
            input = data_transfrom(input)
        input = np.array(input).astype('uint8')
        input = img_transform(input)
        input = input.unsqueeze(0)
        inputs = input if i == 0 else paddle.concat((inputs, input), 0)
    return inputs


def image_output(output_path, images_1, images_2, save_name='result'):
    path = output_path + '/' + save_name + '.png'
    output = paddle.concat((images_1, images_2), 0)
    output = (output + 1.0) / 2.0
    save_image(output, path,nrow=images_1.shape[0], padding=0, normalize=True)

def image_concat(input_path_1,input_path_2,output_path,save_name='result',data_transforms=data_transforms):
    """

    :param input_path_1: 合并的图片的第一个文件夹路径
    :param input_path_2: 合并的图片的第二个文件夹路径
    :param output_path: 输出合并图片的路径
    :param save_name: 保存图片的名称，默认result
    :param data_transforms: 图片大小的更改
    :return: None，结果只有保存图片，无输出结果

    """
    images1=image_dataloader(input_path_1,data_transfrom=data_transforms)
    images2=image_dataloader(input_path_2,data_transfrom=data_transforms)

    image_output(output_path,images1,images2,save_name=save_name)

if __name__ == '__main__':

    input1_path = 'D:\Desktop\plan\Old2Life\output\origin'
    input2_path = 'D:\Desktop\plan\Old2Life\output\\restored_image'
    output_path = 'D:\Desktop\plan\Old2Life\output'

    image_concat(input1_path,input2_path,output_path,save_name='result')

