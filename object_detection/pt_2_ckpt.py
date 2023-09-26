import torch
import mindspore

pth_saved_path = r'D:\Files\GitHub\Utils\object_detection\model_pt\yolov5s.pth'
ckpt_path = r'D:\Files\GitHub\Utils\object_detection\model_ckpt\yolov5s_ascend_v190_coco2017_official_cv_mAP36.9.ckpt'

static_dict_pth = torch.load(pth_saved_path, map_location='cpu')
static_dict_ckpt = mindspore.load_checkpoint(ckpt_path)
print(static_dict_ckpt.keys())
print(static_dict_pth.keys())

length_ckpt = len(static_dict_ckpt)
length_pth = len(static_dict_pth)

print(length_ckpt, length_pth)

ckpt_list = list(static_dict_ckpt.keys())
pth_list = list(static_dict_pth.keys())

ckpt_list_txt_path = r'./ckpt_list.txt'
ckpt_list_txt = open(ckpt_list_txt_path, 'w')


for key in ckpt_list:
    str_key = key + '\r'
    ckpt_list_txt.write(str_key)
    
pth_list_txt_path = r'./pth_list.txt'
pth_list_txt = open(pth_list_txt_path, 'w')
for key in pth_list:
    str_key = key + '\r'
    pth_list_txt.write(str_key)
