import os
import logging
import errno
import time
import datetime
import json
import numpy as np
from PIL import Image

from skimage import io

import tqdm
import cv2
import copy
from shapely.geometry import Polygon
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import seaborn as sns


def my_mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_colored_mask(mask, save_path):
    colormap = [0, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0]
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    lbl_pil.putpalette(colormap)
    lbl_pil.save(save_path)
    return None


# def unclip(box, unclip_ratio=1.25):
#     poly = Polygon(box)
#     if poly.length < 0.01:
#         return None
#     distance = poly.area * unclip_ratio / poly.length
#     offset = pyclipper.PyclipperOffset()
#     offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
#     expanded = np.array(offset.Execute(distance))
#     return expanded


def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def cal_four_para_bbox(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    return int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def cls_by_img(boxes_results):
    det_results = {}
    for index, boxes in enumerate(boxes_results):
        content = boxes['bbox'] + [boxes['score']] + [index]
        if boxes['image_id'] not in det_results:
            det_results[boxes['image_id']] = {boxes['category_id']: [content]}
        elif boxes['category_id'] not in det_results[boxes['image_id']].keys():
            det_results[boxes['image_id']][boxes['category_id']] = [content]
        else:
            det_results[boxes['image_id']][boxes['category_id']].append(content)
    return det_results


def draw_distribution_histogram(nums, path):
    """

    bins: 设置直方图条形的数目
    is_hist: 是否绘制直方图
    is_kde: 是否绘制核密度图
    is_rug: 是否绘制生成观测数值的小细条
    is_vertical: 如果为True，观察值在y轴上
    is_norm_hist: 如果为True，直方图高度显示一个密度而不是一个计数，如果kde设置为True，则此参数一定为True
    """
    plt.figure()
    sns.set()  # 切换到sns的默认运行配置
    plt.hist(x=nums, bins=2000,
             color="steelblue",
             edgecolor="blue")

    # 添加x轴和y轴标签
    plt.xlabel("XXX")
    plt.ylabel("YYY")
    plt.xlim(xmin=0.01, xmax=0.1)

    # 添加标题
    plt.title("Distribution")
    plt.tight_layout()  # 处理显示不完整的问题
    plt.savefig(path, dpi=300)


score_threshold = 0.05

input_json_path = r'D:\Files\GitHub\Utils\temp\test.bbox.json'
output_json_path = input_json_path.replace('.json', '_score_{}.json'.format(score_threshold))
score_hist_path = input_json_path.replace('.json', '.png')
output_score_hist_path = input_json_path.replace('.json', '_score_{}.png'.format(score_threshold))

with open(input_json_path, 'r', encoding='utf-8') as f:
    boxes_results = json.load(f)
# print(boxes_results.keys()[:10])
# raise Nones
# det_results = cls_by_img(boxes_results)
# print(det_results.keys())
# raise None

save_boxes_results = list()
score_list = list()
output_score_list =list()
for item in boxes_results:
    # print(item)
    score_list.append(item['score'])
    if item['score'] < score_threshold:
        continue
    save_boxes_results.append(item)
    output_score_list.append(item['score'])

print('det result filter done! \n'
      'filter threshold is           : {}\n'
      'Number of det before filtering: {}\n'
      'Number of det after  filtering: {}\n'
      'Number of filtering           : {}\n'
      .format(score_threshold, len(boxes_results), len(save_boxes_results), len(boxes_results) - len(save_boxes_results)))

with open(output_json_path,'w') as f:
    json.dump(save_boxes_results,f)
draw_distribution_histogram(score_list,score_hist_path,)
draw_distribution_histogram(output_score_list,output_score_hist_path,)
