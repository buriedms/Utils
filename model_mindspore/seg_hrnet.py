"""
MIT License
Copyright (c) 2019 Microsoft
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import logging

import mindspore
import mindspore.nn as nn
from mindspore import ops

from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import Normal
import mindspore.common.dtype as mstype

import numpy as np

logger = logging.getLogger('hrnet_backbone')

__all__ = ['hrnet18', 'hrnet32', 'hrnet48']

model_urls = {
    # all the checkpoints come from https://github.com/HRNet/HRNet-Image-Classification
    'hrnet18': 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    'hrnet32': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ',
    'hrnet48': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
}


class Softmax(nn.Cell):
    def __init__(self, axis=-1):
        super(Softmax, self).__init__()
        self.axis = axis
        self.max = P.ReduceMax(keep_dims=True)
        self.sum = P.ReduceSum(keep_dims=True)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.div = P.RealDiv()
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float32)
        x = self.sub(x, self.max(x, self.axis))
        x = self.div(self.exp(x), self.sum(self.exp(x), self.axis))
        return x


class Unuse(nn.Cell):
    """Unuse function"""

    def construct(self, x):
        return x


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.SequentialCell(
            nn.BatchNorm2d(num_features, use_batch_statistics=True, **kwargs),
            nn.LeakyReLU(alpha=0.01)
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d


class SpatialGather_Module(nn.Cell):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.softmax = Softmax(axis=2)
        self.transpose = P.Transpose()
        self.expand_dims = P.ExpandDims()
        self.matmul=self.batch_mat_mul

    def batch_mat_mul(self, x, y):
        x = self.cast(x, mstype.float16)
        y = self.cast(y, mstype.float16)
        out = ops.BatchMatMul()(x, y)
        out = self.cast(out,mstype.float32)
        return out

    def construct(self, feats, probs):
        batch_size, c, h, w = probs.shape
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.shape[1], -1)
        feats = self.transpose(feats, (0, 2, 1))  # batch x hw x c
        probs = self.softmax(self.scale * probs)  # batch x k x hw
        # print(probs.shape,feats.shape)
        ocr_context = self.transpose(self.matmul(probs, feats), (0, 2, 1))  # .unsqueeze(3)  # batch x k x c
        ocr_context = self.expand_dims(ocr_context, 3)
        return ocr_context


class _ObjectAttentionBlock(nn.Cell):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''

    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.SequentialCell(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.SequentialCell(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.SequentialCell(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.SequentialCell(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0, has_bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.softmax = Softmax(axis=-1)
        self.matmul = self.batch_mat_mul

    def batch_mat_mul(self, x, y):
        x = self.cast(x, mstype.float16)
        y = self.cast(y, mstype.float16)
        out = ops.BatchMatMul()(x, y)
        out = self.cast(out, mstype.float32)
        return out

    def construct(self, x, proxy):
        batch_size, _, h, w = x.shape
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = self.transpose(query, (0, 2, 1))
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = self.transpose(value, (0, 2, 1))

        sim_map = self.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = self.softmax(sim_map)

        # add bg context ...
        context = self.matmul(sim_map, value)
        context = self.transpose(context, (0, 2, 1))  # .contiguous()
        context = context.view(batch_size, self.key_channels, *x.shape[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = P.ResizeBilinear(size=(h, w), align_corners=True)(context)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Cell):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.SequentialCell(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, has_bias=True),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout(1 - dropout)
        )

    def construct(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(ops.Concat(axis=1)([context, feats]))

        return output


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, pad_mode='pad', group=groups, has_bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, has_bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes, use_batch_statistics=True)
        self.relu = nn.LeakyReLU(alpha=0.01)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes, use_batch_statistics=True)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width, use_batch_statistics=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width, use_batch_statistics=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion, use_batch_statistics=True)
        self.relu = nn.LeakyReLU(alpha=0.01)
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Cell):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True, norm_layer=None):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.LeakyReLU(alpha=0.01)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
            self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                self.norm_layer(num_channels[branch_index] * block.expansion, use_batch_statistics=True),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample, norm_layer=self.norm_layer))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index], norm_layer=self.norm_layer))

        return nn.SequentialCell(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.CellList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.SequentialCell(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  has_bias=False),
                        self.norm_layer(num_inchannels[i], use_batch_statistics=True)))
                elif j == i:
                    fuse_layer.append(Unuse())  # todo None to Unuse()
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.SequentialCell(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=False),
                                self.norm_layer(num_outchannels_conv3x3, use_batch_statistics=True)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.SequentialCell(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=False),
                                self.norm_layer(num_outchannels_conv3x3, use_batch_statistics=True),
                                nn.LeakyReLU(alpha=0.01)))
                    fuse_layer.append(nn.SequentialCell(conv3x3s))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def construct(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]

                    y = y + P.ResizeBilinear(size=(height_output, width_output), align_corners=True)(self.fuse_layers[i][j](x[j]))
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Cell):

    def __init__(self,
                 cfg,
                 norm_layer=None):
        super(HighResolutionNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        # stem network
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               pad_mode='pad', has_bias=False)
        self.bn1 = self.norm_layer(64, use_batch_statistics=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               pad_mode='pad', has_bias=False)
        self.bn2 = self.norm_layer(64, use_batch_statistics=True)
        self.relu = nn.LeakyReLU(alpha=0.01)

        # stage 1
        self.stage1_cfg = cfg['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # stage 2
        self.stage2_cfg = cfg['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = cfg['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = cfg['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        MID_CHANNELS = 512
        KEY_CHANNELS = 256
        last_inp_channels = int(np.sum(pre_stage_channels))
        ocr_mid_channels = MID_CHANNELS
        ocr_key_channels = KEY_CHANNELS

        self.conv3x3_ocr = nn.SequentialCell(
            nn.Conv2d(last_inp_channels, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1, pad_mode='pad', has_bias=True),
            norm_layer(ocr_mid_channels, use_batch_statistics=True),
            nn.LeakyReLU(alpha=0.01),
        )
        self.ocr_gather_head = SpatialGather_Module(10)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )
        self.cls_head = nn.Conv2d(
            ocr_mid_channels, 4, kernel_size=1, stride=1, padding=0, has_bias=True)

        self.aux_head = nn.SequentialCell(
            nn.Conv2d(last_inp_channels, last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            norm_layer(last_inp_channels, use_batch_statistics=True),
            nn.LeakyReLU(alpha=0.01),
            nn.Conv2d(last_inp_channels, 4,
                      kernel_size=1, stride=1, padding=0, has_bias=True)
        )
        self.param_init()

    def param_init(self):
        cnt = 0
        for _, cell in self.cells_and_names():
            # print(cell)
            if isinstance(cell, nn.BatchNorm2d):
                cnt += 1
                # print(cell)
                cell.requires_grad = True
                cell.moving_mean.requires_grad = True
                cell.moving_variance.requires_grad = True
        # print('the number of bn:\t{}'.format(cnt))

    def _make_transition_layer(
        self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.SequentialCell(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  pad_mode='pad',
                                  has_bias=False),
                        self.norm_layer(num_channels_cur_layer[i], use_batch_statistics=True),
                        nn.LeakyReLU(alpha=0.01)))
                else:
                    transition_layers.append(Unuse())  # todo None to Unuse()
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.SequentialCell(
                        nn.Conv2d(
                            inchannels, outchannels, kernel_size=3, stride=2, padding=1, pad_mode='pad', has_bias=False),
                        self.norm_layer(outchannels, use_batch_statistics=True),
                        nn.LeakyReLU(alpha=0.01)))
                transition_layers.append(nn.SequentialCell(conv3x3s))

        return nn.CellList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                self.norm_layer(planes * block.expansion, use_batch_statistics=True),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, norm_layer=self.norm_layer))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=self.norm_layer))

        return nn.SequentialCell(layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output,
                                     norm_layer=self.norm_layer)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.SequentialCell(modules), num_inchannels

    def construct(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []

        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []

        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        outputs = {}
        # See note [TorchScript super()]
        outputs['res2'] = x[0]  # 1/4
        outputs['res3'] = x[1]  # 1/8
        outputs['res4'] = x[2]  # 1/16
        outputs['res5'] = x[3]  # 1/32
        _, _, x0_h, x0_w = x[0].shape
        interp = P.ResizeBilinear(size=(x0_h, x0_w), align_corners=True)
        x1 = interp(x[1])
        x2 = interp(x[2])
        x3 = interp(x[3])

        feats = ops.Concat(axis=1)([x[0], x1, x2, x3])

        out_aux_seg = []

        # ocr
        out_aux = self.aux_head(feats)
        # compute contrast feature
        feats = self.conv3x3_ocr(feats)

        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)

        out = self.cls_head(feats)

        out_aux_seg.append(out_aux)
        out_aux_seg.append(out)

        return out_aux_seg


def _hrnet(arch, pretrained, progress, **kwargs):

    from .hrnet_config import MODEL_CONFIGS

    #     from .hrnet_config import MODEL_CONFIGS
    # except ImportError:
    #     from .hrnet_config import MODEL_CONFIGS # todo need change 1
    model = HighResolutionNet(MODEL_CONFIGS[arch], **kwargs)
    if pretrained:
        # if int(os.environ.get("mapillary_pretrain", 0)):
        #     logger.info("load the mapillary pretrained hrnet-w48 weights.")
        #     model_url = model_urls['hrnet48_mapillary_pretrain']
        # else:
        #     model_url = model_urls[arch]

        import mindspore
        pretrained_dict = mindspore.load_checkpoint('./pretrained/hrnetv2_w{}_imagenet_pretrained.ckpt'.format(arch[-2:]))
        # pretrained_dict = mindspore.load('./checkpoints/hrnetv2_w{}_imagenet_pretrained.ckpt'.format(arch[-2:])) # todo need change 2
        print('=> loading pretrained model {}'.format(pretrained))
        model_dict = model.parameters_dict()
        # print(model_dict)
        # raise None
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict.keys()}
        # for k, _ in pretrained_dict.items():
        # print('=> loading {} pretrained model {}'.format(k, pretrained))
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)
        mindspore.load_param_into_net(model, pretrained_dict)
        # print(model.conv1.weight[0,0])
    return model


def hrnet18(pretrained=False, progress=True, **kwargs):
    r"""HRNet-18 model
    """
    return _hrnet('hrnet18', pretrained, progress,
                  **kwargs)


def hrnet32(pretrained=False, progress=True, **kwargs):
    r"""HRNet-32 model
    """
    return _hrnet('hrnet32', pretrained, progress,
                  **kwargs)


def hrnet48(pretrained=False, progress=True, **kwargs):
    r"""HRNet-48 model
    """
    return _hrnet('hrnet48', pretrained, progress,
                  **kwargs)


if __name__ == '__main__':
    model = hrnet32(pretrained=False)
    # print(model)
    import numpy as np
    from mindspore import Tensor
    from mindspore import dtype as msdtype

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)

    data = np.random.randn(2, 3, 512, 512)
    input = Tensor(data, dtype=msdtype.float32)
    state_dict = model.parameters_dict()
    log_path = './hrnet32_param.log'
    logger = open(log_path, 'w')
    logger.write('The number of all parameter : \t {}\n'.format(len(state_dict)))
    names = sorted(list(state_dict.keys()))
    params = list(state_dict[key] for key in names)
    items = zip(names, params)
    for i, item in enumerate(items):
        logger.write('{}\n'.format(item))
    logger.close()
    # print(model)
    output = model(input)
    print(len(output))
    print(output[0].shape)
    print(output[1].shape)
