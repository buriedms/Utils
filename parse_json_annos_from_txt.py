import os
import datetime
import json


#
# def create_train_data_from_txt(image_dir, anno_path):
#     """Filter valid image file, which both in image_dir and anno_path."""
#
#     now = datetime.datetime.now()
#
#     data = dict(
#         info=dict(
#             description=None,
#             url=None,
#             version=None,
#             year=now.year,
#             contributor=None,
#             date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
#         ),
#         licenses=[dict(url=None, id=0, name=None, )],
#         images=[
#             # license, url, file_name, height, width, date_captured, id
#         ],
#         type="instances",
#         annotations=[
#             # segmentation, area, iscrowd, image_id, bbox, category_id, id
#         ],
#         categories=[
#             # supercategory, id, name
#         ],
#     )
#     # ========== you can change
#     labels = ['Intersection_1', 'Intersection_2']
#     class_name_to_id = {}
#
#     for class_id, label in enumerate(labels):
#         data["categories"].append(
#             dict(supercategory=None, id=class_id, name=label, )
#         )
#         class_name_to_id[label] = class_id
#
#     def anno_parser(annos_str):
#         """Parse annotation from string to list."""
#         annos = []
#         for anno_str in annos_str:
#             anno = anno_str.strip().split(",")
#             xmin, ymin, xmax, ymax = list(map(float, anno[:4]))
#             if xmin == 0 and ymin == 0 and xmax == 1024 and ymax == 1024:
#                 continue
#             cls_id = int(anno[4])
#             iscrowd = int(anno[5])
#             annos.append([xmin, ymin, xmax, ymax, cls_id, iscrowd])
#         return annos
#
#     image_files = []
#     file_names = []
#     image_anno_dict = {}
#     if not os.path.isdir(image_dir):
#         raise RuntimeError("Path given is not valid.")
#     if not os.path.isfile(anno_path):
#         raise RuntimeError("Annotation file is not valid.")
#
#     with open(anno_path, "rb") as f:
#         lines = f.readlines()
#     for line in lines:
#         line_str = line.decode("utf-8").strip()
#         line_split = str(line_str).split(' ')
#         file_name = line_split[0]
#         image_path = os.path.join(image_dir, file_name)
#         file_name = os.path.basename(file_name)
#         if os.path.isfile(image_path):
#             image_anno_dict[file_name] = anno_parser(line_split[1:])
#             image_files.append(image_path)
#             file_names.append(file_name)
#     for image_id, file_name in enumerate(file_names):
#
#         file_path = os.path.join(image_dir, file_name)
#         file_name = os.path.basename(file_name)
#         height = 1024
#         width = 1024
#
#         data["images"].append(
#             dict(
#                 license=0,
#                 url=None,
#                 file_name=file_name,
#                 file_path=file_path,
#                 height=height,
#                 width=width,
#                 date_captured=None,
#                 id=image_id,
#             )
#         )
#         annos = image_anno_dict[file_name]
#         for anno_idx, anno in enumerate(annos):
#             xmin, ymin, xmax, ymax, cls_id, iscrowd = anno
#             # print('anno:', anno)
#             # raise None
#             # cls_id = class_name_to_id[cls_name]
#             data["annotations"].append(
#                 dict(
#                     id=len(data["annotations"]),
#                     image_id=image_id,
#                     category_id=cls_id,
#                     segmentation=segmentations[instance],
#                     area=area,
#                     bbox=bbox,
#                     iscrowd=0,
#                 )
#             )
#
#     print(image_files)
#     image = dict()
#     annotations = dict()
#     # return image_files, image_anno_dict

def parse_json_annos_from_txt(anno_file, output_path):
    """for user defined annotations text file, parse it to json format data"""
    if not os.path.isfile(anno_file):
        raise RuntimeError("Evaluation annotation file {} is not valid.".format(anno_file))

    annos = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # set categories field
    labels = ['Intersection_1', 'Intersection_2']
    for i, cls_name in enumerate(labels):
        annos["categories"].append({"id": i + 1, "name": cls_name})

    with open(anno_file, "rb") as f:
        lines = f.readlines()

    img_id = 1
    anno_id = 1
    for line in lines:
        line_str = line.decode("utf-8").strip()
        line_split = str(line_str).split(' ')
        # set image field
        file_name = line_split[0].replace('fusai_road/', '')
        annos["images"].append({"file_name": file_name, "id": img_id})
        # set annotations field
        for anno_info in line_split[1:]:
            anno = anno_info.split(",")
            x = float(anno[0])
            y = float(anno[1])
            w = float(anno[2]) - float(anno[0])
            h = float(anno[3]) - float(anno[1])
            category_id = int(anno[4])
            iscrowd = int(anno[5])
            annos["annotations"].append({"bbox": [x, y, w, h],
                                         "area": w * h,
                                         "category_id": category_id,
                                         "iscrowd": iscrowd,
                                         "image_id": img_id,
                                         "id": anno_id})
            anno_id += 1
        img_id += 1
    print(annos)
    with open(output_path, "w") as f:
        json.dump(annos, f)

    return annos


# path_img = r'E:\datasets\hwst'
path_anno = r'D:\Files\GitHub\faster_rcnn_new\rsipac.txt'
output_path = r'./temp/fusai_rsipic.json'
# create_train_data_from_txt(path_img, path_anno)
parse_json_annos_from_txt(path_anno, output_path)
