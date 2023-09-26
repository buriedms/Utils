"""
这个脚本整忘记了

"""

# author scu cj
# 输入: coco格式目标检测数据集
# 输出: coco格式目标检测数据集
# parameters
# image_width：裁切后的图像宽度
# image_height: 裁切后的图像高度
# img_overlap: 裁切时重叠像素数
# iou_thres: 裁切后图像块中的目标框占真值目标框比例大于多少时才判定裁切图像块中有目标并进行标注（一般来说目标检测框IOU大于0.25才能算）

import os
import numpy as np
import cv2
from natsort import natsorted
from requests import patch
from sklearn import neighbors
from tqdm import tqdm
import json
from skimage import measure
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

datasrc_dir = "/mnt/datas4t/PlaneRecogData/data/val"
# label_file = "/mnt/datas4t/PlaneRecogData/data"
datadst_dir = "/mnt/datas4t/PlaneRecogData/data_patch/val"
dst_labeljson_file = "/mnt/datas4t/PlaneRecogData/data_patch/instances_val.json"
coco_label = dict()

images = []
annotations = []
categories = {}

image_width = 512
image_height = 512
img_overlap = 128
iou_thres = 0.25

coco_label["info"] = {
    "image_width": 512,
    "image_height": 512,
    "patch_overlap": 128,
    "iou_thres": 0.25
}

cat_name = "ABCDEFGHIJK"
cat_id = range(1, len(cat_name) + 1)
categories = [{"id": i, "name": j} for i, j in zip(cat_id, cat_name)]

img_id = 1
ann_id = 1
img = {}
ann = {}
files = [i for i in os.listdir(datasrc_dir) if os.path.isfile(datasrc_dir + "/" + i)]
files = natsorted(files)  # 获取所有标注数据
root = datasrc_dir
with tqdm(range(0, len(files), 2)) as pbar:
    for prno in range(0, len(files), 2):
        imagepath = root + "/" + files[prno + 1]
        labelpath = root + "/" + files[prno]
        ori_image = cv2.imread(imagepath)
        w, h, _ = ori_image.shape
        with open(labelpath, "r") as jsonfile:
            labeljson = json.load(jsonfile)  # labeljson是大图的标注数据
        # 构造一个二值标注图
        labelimage = np.zeros(ori_image.shape)  # .astype(int)     # labelimage是大图的实例标注图
        semantic_label = np.zeros(ori_image.shape)  # .astype(int) # semantic_label是大图的语义标注图
        for i1 in range(len(labeljson["shapes"])):  # 遍历大图中的每一个实例
            pt = labeljson["shapes"][i1]  # 实例点标注
            labelimage = cv2.fillPoly(labelimage, np.array([pt["points"]]).astype(int), (i1 + 1, i1 + 1, i1 + 1))  # 不同的实例有不一样的框
            # plt.title("labelimage")
            # plt.imshow(labelimage)
            # plt.show()
            cat_id = ord(pt["label"]) - ord('A') + 1  # 实例对应类型标注
            semantic_label = cv2.fillPoly(semantic_label, np.array([pt["points"]]).astype(int), (cat_id, cat_id, cat_id))  # 不同的实例有不一样的框

        # print(labelpath)
        objid, idcount = np.unique(labelimage, return_counts=True)  # 统计实例占像素数
        # print(objid)
        # print(idcount)
        labelimage_statics = dict(zip(objid, idcount))
        # print(labelimage_statics)
        labelimage_statics.pop(0)  # 去掉背景的统计

        # 裁切图像

        for sx in range(0, w, image_width - img_overlap):
            for sy in range(0, h, image_height - img_overlap):
                if sx + image_width < w:
                    ex = sx + image_width
                else:
                    ex = w
                    sx = w - image_width
                if sy + image_height < h:
                    ey = sy + image_height
                else:
                    sy = h - image_height
                    ey = h

                patch_file_name = datadst_dir + "/" + str(img_id) + ".jpg"
                img["file_name"] = patch_file_name
                img["height"] = ey - sy
                img["width"] = ex - sx
                img["id"] = img_id
                img_id = img_id + 1
                labelpatch = labelimage[sy:ey, sx:ex, :]
                patchimg = ori_image[sy:ey, sx:ex, :]

                if labelpatch.max() == 0:
                    # 说明没有目标,label的最大值等于0，说明没有目标,直接保存而不加标签
                    cv2.imwrite(patch_file_name, patchimg)
                    images.append(img)
                    img = {}
                else:
                    # 否则计算patch中的存在的目标实例的框占整体标注框的比例,如果比例大于0.5就认为是一个真值,并进行标注
                    # objlabel是图像中实例的编号, labelarea是统计这个patch中目标旋转框实例像素有多少
                    objlabel, labelarea = np.unique(labelpatch, return_counts=True)  # 要去掉0
                    patch_statics = dict(zip(objlabel, labelarea))
                    patch_statics.pop(0)
                    # 计算每个实例占总的比例
                    for k, v in patch_statics.items():
                        # print("k:"+str(k))
                        # print("v:"+str(v))
                        if (v / labelimage_statics[k]) < iou_thres:
                            continue
                        # 小于阈值则不管他，否则作为一个实例框的标注
                        # 先获取该实例的类别
                        semantic_patch = semantic_label[sy:ey, sx:ex, :]
                        cat_id = semantic_patch[labelpatch == k][0]
                        ann["id"] = ann_id
                        ann_id += 1
                        ann["image_id"] = img["id"]
                        ann["category_id"] = cat_id
                        tempmat = np.copy(labelpatch)
                        tempmat = np.where(tempmat != k, 0, 255)
                        # plt.figure()
                        # plt.title("patchimg")
                        # plt.imshow(patchimg)
                        # plt.figure()
                        # plt.title("labelpatch")
                        # plt.imshow(labelpatch)
                        # plt.figure()
                        # plt.title("temp")
                        # plt.imshow(tempmat)
                        # plt.show()
                        labeled_img = label(tempmat, connectivity=1, background=0, return_num=False)
                        annprops = regionprops(labeled_img)[0]
                        annsy, annsx, _, anney, annex, _ = annprops.bbox  # 因为没有去除通道维度
                        ann["bbox"] = [annsx, annsy, annex - annsx, anney - annsy]
                        # tempmat = cv2.rectangle(tempmat.astype(np.uint8),(23,24),(123,124),(255,0,0),5)
                        # tempmat = cv2.rectangle(patchimg.astype(np.uint8),(annsx,annsy),(annex,anney),(233,0,0),2)
                        # plt.figure()
                        # plt.title("tempmat")
                        # plt.imshow(tempmat)
                        # plt.show()
                        ann["area"] = annprops.area_bbox
                        ann["iscrowd"] = 0
                        annotations.append(ann)
                        ann = {}
                    cv2.imwrite(patch_file_name, patchimg)
                    images.append(img)
                    img = {}
        pbar.update(1)

coco_label["images"] = images
coco_label["annotations"] = annotations
coco_label["categories"] = categories

with open(dst_labeljson_file, "w") as fobj:
    json.dump(coco_label, fobj)
