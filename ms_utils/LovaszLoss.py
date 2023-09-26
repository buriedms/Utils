# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import mindspore
from mindspore import Tensor, ops
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import stop_gradient


class SoftmaxCrossEntropyLoss(nn.Cell):
    def __init__(self, num_cls=21, ignore_label=255, weight=None):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        if weight is None:
            weight = [1., 1., 1., 1.]
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.greater_equal = P.GreaterEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.zeros_like = P.ZerosLike()
        self.maskedfill = P.MaskedFill()
        self.equal = P.Equal()
        self.logical_or = P.LogicalOr()
        self.logical_and = P.LogicalAnd()
        self.less = P.Less()
        self.weight_list = weight

    def construct(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, self.num_cls))
        # weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(self.zeros_like(labels_int), mstype.float32)
        for label, weight_value in enumerate(self.weight_list):
            masks = self.equal(labels_int, label)
            weights = self.maskedfill(weights, masks, weight_value)
        masks = self.logical_or(self.less(labels_int, 0), self.equal(labels_int, self.ignore_label))
        weights = self.maskedfill(weights, masks, 0.)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        masks = self.logical_and(self.greater_equal(labels_int, 0), self.not_equal(labels_int, self.ignore_label))
        loss = self.div(self.sum(loss), self.sum(self.cast(masks, mstype.float32)))
        return loss


class LovaszLoss(nn.Cell):
    def __init__(self, num_cls=21,ignore_label=255, need_softmax=False):
        """
        :param num_cls: Not used
        :param ignore_label: Not used
        :param need_softmax: Whether softmax is required for normalization
        """
        super(LovaszLoss, self).__init__()

        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.need_softmax = need_softmax
        self.softmax = ops.Softmax(axis=1)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.sort = ops.Sort(axis=0, descending=True)
        self.equal = ops.Equal()
        self.concat = ops.Concat(axis=0)
        self.tensor_dot = ops.tensor_dot
        self.mean = ops.ReduceMean()
        self.sub = ops.Sub()
        self.zeros = ops.Zeros()
        self.print = ops.Print()
        self.slice = ops.Slice()


    def flatten_probas(self, probas: Tensor, labels: Tensor):
        """
        Flattens predictions in the batch
        """
        if probas.ndim == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.shape
            probas = self.reshape(probas, (B, 1, H, W))
        B, C, H, W = probas.shape
        probas = self.reshape(P.Transpose()(probas, (0, 2, 3, 1)), (-1, C))  # B * H * W, C = P, C
        labels = self.reshape(labels, (-1,))
        return probas, labels

    def lovasz_grad(self, gt_sorted: Tensor):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = gt_sorted.shape[0]
        gts = gt_sorted.sum()
        # intersection = gts - self.cast(gt_sorted, mstype.float32).cumsum(0)
        # union = gts + self.cast((1 - gt_sorted), mstype.float32).cumsum(0)

        # jaccard = 1. - intersection / union
        jaccard = 1. - (gts - gt_sorted.cumsum(0)) / (gts + (1. - gt_sorted).cumsum(0))
        a = self.slice(jaccard,(0,),(p-1,))
        print(a.shape)
        # jaccard = self.reshape(jaccard,(-1,1))
        # a = mindspore.numpy.roll(jaccard, 1, 0)
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
            # slice_1 = slice(1, p)
            # slice_2 = slice(0, - 1)
            # a = self.gather(jaccard, ops.LinSpace()(self.cast(1, mstype.float32), self.cast(p, mstype.float32), 1), 0)
            # b = self.gather(jaccard, range(1, p), 0)
            # temp = self.sub(jaccard[slice_1].copy(), jaccard[slice_2].copy())
            # print('st9')
            # jaccard[slice_1] = jaccard[slice_1] - jaccard[slice_2]
        # jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def lovasz_softmax_flat(self, probas: Tensor, labels: Tensor, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
          labels: [P] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        losses = []
        C = probas.shape[1]
        for c in range(C):
            fg = self.cast((self.equal(labels, c)), mstype.float32)  # foreground for class c
            # if (classes == 'present' and fg.sum() == 0):
            #     continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                # class_pred = probas[:, 0]
                class_pred = self.gather(probas, self.cast(0, mstype.int32), 1)
            else:
                # class_pred = probas[:, c]
                class_pred = self.gather(probas, self.cast(c, mstype.int32), 1)
            # class_pred = probas[:, c]

            fg = stop_gradient(fg)
            errors = (fg - class_pred).abs()
            errors_sorted, perm = self.sort(errors)

            fg_sorted = self.gather(fg, perm, 0)
            losses.append(self.tensor_dot(errors_sorted, stop_gradient(self.lovasz_grad(fg_sorted)), 1).expand_dims(0))
        losses = self.concat(losses)
        return self.mean(losses, 0)

    def construct(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
          probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                  Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
          labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
          classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
          per_image: compute the loss per image instead of per batch
          ignore: void class labelas
        """
        if self.need_softmax:
            probas = self.softmax(probas)
        loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels), classes=classes)
        return loss

    # def construct(self, gt_sorted: Tensor):
    #     return self.lovasz_grad(gt_sorted)


if __name__ == '__main__':
    import numpy as np
    from mindspore import ops as P

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE, device_target="CPU")
    pred_shape = [2, 4, 1024, 1024]
    label_shape = [2, 1024, 1024]
    np.random.seed(100)
    preds = Tensor(np.random.randn(*pred_shape), dtype=mindspore.float32)
    np.random.seed(1000)
    labels = Tensor(np.random.randint(0, 4, label_shape), dtype=mindspore.int32)
    loss_func = LovaszLoss(num_cls=2,ignore_label=255, need_softmax=True)
    softmax = P.Softmax(axis=1)
    loss = loss_func(preds, labels)
    print(loss)
    # input1 = Tensor(np.random.randn(5520, ), mstype.float32)
    # output1 = loss_func(input1)
    # print(output1.shape)
