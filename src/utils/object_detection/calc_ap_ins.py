import numpy as np
import json
from numpy.core.numeric import indices
from skimage.draw import polygon
from tqdm import tqdm


def mask2bbox(mask):
    '''
    Input
        type : numpy array
        example :
        mask = [
            [x1, y1],
            [x2, y2],
            [x3, y3],
            ...
            [xn, yn]
        ]

    Output
        type : list
        example : [Xmin, Ymin, Xmax, Ymax]
    '''
    Xmin, Xmax = np.min(mask[:, 0]), np.max(mask[:, 0])
    Ymin, Ymax = np.min(mask[:, 1]), np.max(mask[:, 1])

    return [Xmin, Ymin, Xmax, Ymax]


def clear_boundary(polygon, Xmax, Ymax):
    polygon[:, 0] = np.clip(polygon[:, 0], 0, Xmax)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, Ymax)

    return polygon


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mask_iou(maskA, maskB):
    union = np.logical_or(maskA, maskB).astype('int')
    intersection = np.logical_and(maskA, maskB).astype('int')
    iou = intersection.sum() / union.sum()
    return iou


def polygon_iou(polyA, polysB):
    '''
    polyA : 1 pred polygon
    polysB : all polygons in matched image & class
    '''
    pred_poly = np.array(polyA, dtype=np.int0).reshape(-1, 2)
    ious = []
    for gt_poly in polysB:
        gt_poly = np.array(gt_poly, dtype=np.int0).reshape(-1, 2)

        all_x_coords = np.concatenate([pred_poly[:, 0], gt_poly[:, 0]])
        all_y_coords = np.concatenate([pred_poly[:, 1], gt_poly[:, 1]])
        XMin, XMax = np.min(all_x_coords), np.max(all_x_coords)
        YMin, YMax = np.min(all_y_coords), np.max(all_y_coords)
        height, width = YMax - YMin, XMax - XMin

        # Shift poly(for minimal background image)
        pred_poly[:, 0] -= XMin
        pred_poly[:, 1] -= YMin
        gt_poly[:, 0] -= XMin
        gt_poly[:, 1] -= YMin

        # Background
        pred_mask = np.zeros((height + 1, width + 1), dtype=np.int0)
        gt_mask = np.zeros((height + 1, width + 1), dtype=np.int0)
        # fill poly
        prow, pcol = polygon(pred_poly[:, 1], pred_poly[:, 0], pred_mask.shape)
        grow, gcol = polygon(gt_poly[:, 1], gt_poly[:, 0], pred_mask.shape)
        pred_mask[prow, pcol] = 1
        gt_mask[grow, gcol] = 1

        iou = mask_iou(pred_mask, gt_mask)
        ious.append(iou)

    return ious


def mask_map(answer_dict, pred_dict, exclude_not_in_annotations=False, iou_threshold=0.5, verbose=False):
    '''
    answer_dict = {
        ImageName : {
            ClassName: [
                {
                    "polygon": [x1,y1,x2,y2,x3,y3, ...]

                }
            ]
        }
    }

    pred_dict = {
        ImageName : {
            ClassName : [
                {
                    "polygon" : [x1,y1,x2,y2,x3,y3, ...],
                    "score" : 0.7777
                }
            ]
        }
    }
    '''
    if isinstance(answer_dict, str):
        with open(answer_dict, 'r') as gt_data:
            valid = json.load(gt_data)

    if isinstance(pred_dict, str):
        with open(pred_dict, 'r') as pred_data:
            preds = json.load(pred_data)

    ann_unique = list(set(valid.keys()))
    preds_unique = list(set(preds.keys()))

    if verbose:
        print('Number of files in annotations: {}'.format(len(ann_unique)))
        print('Number of files in predictions: {}'.format(len(preds_unique)))

    if exclude_not_in_annotations:
        preds = {n: preds[n] for n in ann_unique}
        if verbose:
            print('Number of files in detection after reduction: {}'.format(len(preds_unique)))

    unique_classes = []
    for v in valid:
        image_classes = list(valid[v].keys())
        image_classes = [str(x) for x in image_classes]
        unique_classes.extend(image_classes)
    unique_classes = np.array(list(set(unique_classes)))

    average_precisions = {}
    for zz, label in enumerate(sorted(unique_classes)):
        # Negative class
        if str(label) == 'nan':
            continue

        false_positives = []
        true_positives = []
        scores = []
        num_annotations = 0.0

        for i in range(len(ann_unique)):
            detections, annotations = [], []
            id = ann_unique[i]  # ImageID(Name)
            if id in preds:
                if label in preds[id]:
                    detections = preds[id][label]

            if id in valid:
                if label in valid[id]:
                    annotations = valid[id][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue

            num_annotations = len(annotations)

            detected_annotations = []

            annotations = [x['polygon'] for x in annotations]

            for d in tqdm(detections):
                scores.append(d['score'])

                det_poly = d['polygon']

                if len(annotations) == 0:
                    false_positives.append(1)
                    true_positives.append(0)
                    continue

                # IoU
                overlaps = np.array([polygon_iou(det_poly, annotations)])
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives.append(0)
                    true_positives.append(1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives.append(1)
                    true_positives.append(0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        false_positives = np.array(false_positives)
        true_positives = np.array(true_positives)
        scores = np.array(scores)

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations

        if verbose:
            s1 = "{:30s} | {:.6f} | {:7d}".format(label, average_precision, int(num_annotations))
            print(s1)

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / present_classes
    if verbose:
        print('mAP: {:.6f}'.format(mean_ap))
    return mean_ap, average_precisions


det = 'D:/Files/GitHub/Utils/temp/json_dir/preds_coco.json'
gt = 'D:/Files/GitHub/Utils/temp/json_dir/gt_coco.json'
mask_map(gt, det,exclude_not_in_annotations=True, verbose=True)