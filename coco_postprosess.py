import numpy as np
import json


def do_nms_for_results(results,thresh=0.6):
    """
    nms result

    results = {'img_id': {0: [x ,y ,w ,h],1: ,2: ,...}}

    """
    det_boxes = []

    # for box in results:
    #     img_id, bbox, clsi, score = box
    #     bbox = np.array(bbox)
    #     print(bbox)
    #     dets = np.insert(bbox.copy(), -1, score)
    #     keep_index = _diou_nms(dets, thresh=0.6)
    #
    #     keep_box = [{'image_id': int(img_id),
    #                  'category_id': int(clsi),
    #                  'bbox': list(dets[i][:4].astype(float)),
    #                  'score': dets[i][4].astype(float)}
    #                 for i in keep_index]
    #     det_boxes.extend(keep_box)

    for img_id in results:
        for clsi in results[img_id]:
            dets = results[img_id][clsi]
            dets = np.array(dets)
            keep_index = diou_nms(dets, thresh=thresh)

            keep_box = [{'image_id': int(img_id),
                         'category_id': int(clsi),
                         'bbox': list(dets[i][:4].astype(float)),
                         'score': dets[i][4].astype(float)}
                        for i in keep_index]
            det_boxes.extend(keep_box)
    return det_boxes


def diou_nms(dets, thresh=0.5):
    """convert xywh -> xmin ymin xmax ymax"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = x1 + dets[:, 2]
    y2 = y1 + dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # center_x1 = (x1[i] + x2[i]) / 2
        # center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        # center_y1 = (y1[i] + y2[i]) / 2
        # center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
        # inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        # out_max_x = np.maximum(x2[i], x2[order[1:]])
        # out_max_y = np.maximum(y2[i], y2[order[1:]])
        # out_min_x = np.minimum(x1[i], x1[order[1:]])
        # out_min_y = np.minimum(y1[i], y1[order[1:]])
        # outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
        # diou = ovr - inter_diag / outer_diag
        # diou = np.clip(diou, -1, 1)
        # inds = np.where(diou <= thresh)[0]
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def _diou_nms(dets, thresh=0.5):
    """convert xywh -> xmin ymin xmax ymax"""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = x1 + dets[:, 2]
    y2 = y1 + dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # xx1 = np.maximum(x1[i], x1[order[1:]])
        # yy1 = np.maximum(y1[i], y1[order[1:]])
        # xx2 = np.minimum(x2[i], x2[order[1:]])
        # yy2 = np.minimum(y2[i], y2[order[1:]])
        #
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        # inter = w * h
        #
        # ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #
        # inds = np.where(ovr <= thresh)[0]


        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        center_x1 = (x1[i] + x2[i]) / 2
        center_x2 = (x1[order[1:]] + x2[order[1:]]) / 2
        center_y1 = (y1[i] + y2[i]) / 2
        center_y2 = (y1[order[1:]] + y2[order[1:]]) / 2
        inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
        out_max_x = np.maximum(x2[i], x2[order[1:]])
        out_max_y = np.maximum(y2[i], y2[order[1:]])
        out_min_x = np.minimum(x1[i], x1[order[1:]])
        out_min_y = np.minimum(y1[i], y1[order[1:]])
        outer_diag = (out_max_x - out_min_x) ** 2 + (out_max_y - out_min_y) ** 2
        diou = ovr - inter_diag / outer_diag
        # print('diou:',diou)
        diou = np.clip(diou, -1, 1)
        inds = np.where(diou <= thresh)[0]
        order = order[inds + 1]
    return keep


def apply_nms(all_boxes, all_scores, thres, max_boxes):
    """Apply NMS to bboxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算所有box面积

    order = all_scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thres)[0]

        order = order[inds + 1]
    return keep


def cls_by_img(boxes_results):
    det_results = {}
    for boxes in boxes_results:
        if boxes['image_id'] not in det_results:
            # print(boxes['bbox']+[1.])
            # raise None
            det_results[boxes['image_id']] = {boxes['category_id']: [boxes['bbox'] + [boxes['score']]]}
        elif boxes['category_id'] not in det_results[boxes['image_id']]:
            det_results[boxes['image_id']] = {boxes['category_id']: [boxes['bbox'] + [boxes['score']]]}
        else:
            det_results[boxes['image_id']][boxes['category_id']].append(boxes['bbox'] + [boxes['score']])
    return det_results


if __name__ == '__main__':
    json_path = r'./temp/test.bbox.json'
    thresh = 0.01
    save_json_path = r'./thresh_{}_test.bbox.json'.format(thresh)
    f = open(json_path, 'r', encoding='utf-8')
    boxes_results = json.load(f)
    print(len(boxes_results))
    print(boxes_results[:5])
    det_results = cls_by_img(boxes_results)
    print(len(det_results))
    boxes_num = {img: 0 for img in det_results}
    for img in det_results:
        # print(det_results[img])
        for cls in det_results[img]:
            for _ in det_results[img][cls]:
                # print(boxes_num[img])
                boxes_num[img] += 1
    print(boxes_num)

    det_results = do_nms_for_results(det_results,thresh)
    json_str = json.dumps(det_results, indent=4)
    outJson = open(save_json_path, 'w')
    outJson.write(json_str)
    print(len(det_results))
    print(det_results[:5])

    det_results = cls_by_img(det_results)
    print(len(det_results))
    boxes_num = {img: 0 for img in det_results}
    for img in det_results:
        # print(det_results[img])
        for cls in det_results[img]:
            for _ in det_results[img][cls]:
                # print(boxes_num[img])
                boxes_num[img] += 1
    print(boxes_num)
