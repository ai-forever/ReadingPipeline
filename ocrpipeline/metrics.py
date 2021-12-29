import numpy as np
from shapely.geometry import Polygon


def get_accuracy(y_true, y_pred):
    """Calc accuracy between two list of strings."""
    scores = []
    for true, pred in zip(y_true, y_pred):
        scores.append(true == pred)
    avg_score = np.mean(scores)
    return avg_score


def levenshtein_distance(first, second):
    distance = [[0 for _ in range(len(second) + 1)]
                for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        for j in range(len(second) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                diag = distance[i - 1][j - 1] + (first[i - 1] != second[j - 1])
                upper = distance[i - 1][j] + 1
                left = distance[i][j - 1] + 1
                distance[i][j] = min(diag, upper, left)
    return distance[len(first)][len(second)]


def cer(gt_texts, pred_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_chars = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        lev_distances += levenshtein_distance(pred_text, gt_text)
        num_gt_chars += len(gt_text)
    return lev_distances / num_gt_chars


def wer(gt_texts, pred_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_words = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        gt_words, pred_words = gt_text.split(), pred_text.split()
        lev_distances += levenshtein_distance(pred_words, gt_words)
        num_gt_words += len(gt_words)
    return lev_distances / num_gt_words


def iou_bbox(boxA, boxB):
    # source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    # https://stackoverflow.com/a/58108241
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def iou_polygon(polygon1, polygon2):
    polygon1 = Polygon(polygon1)
    polygon2 = Polygon(polygon2)
    if polygon1.is_valid and polygon2.is_valid:
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou = intersect / union
        return iou
    else:
        return 0
