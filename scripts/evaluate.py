import argparse
import os
import json
from tqdm import tqdm
import numpy as np

from ocrpipeline.utils import AverageMeter
from ocrpipeline.metrics import (
    contour2shapely, get_accuracy, cer, wer, iou_polygon
)


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[list_of_numbers[i], list_of_numbers[i+1]]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


def polygon2bbox(polygon):
    x_min = np.inf
    y_min = np.inf
    x_max = -np.inf
    y_max = -np.inf
    for x, y in polygon:
        if x > x_max:
            x_max = x
        if y > y_max:
            y_max = y
        if x < x_min:
            x_min = x
        if y < y_min:
            y_min = y
    return int(x_min), int(y_min), int(x_max), int(y_max)


def class_names2id(class_names, data):
    """Match class names to categoty ids using annotation in COCO format."""
    category_ids = []
    for class_name in class_names:
        for category_info in data['categories']:
            if category_info['name'] == class_name:
                category_ids.append(category_info['id'])
    return category_ids


def get_data_from_image(data, image_id, class_names):
    texts = []
    polygons = []
    category_ids = class_names2id(class_names, data)
    for idx, data_ann in enumerate(data['annotations']):
        if (
            data_ann['image_id'] == image_id
            and data_ann['category_id'] in category_ids
            and data_ann['attributes']
            and data_ann['attributes']['translation']
            and data_ann['segmentation']
        ):
            polygon = numbers2coords(data_ann['segmentation'][0])
            polygons.append(polygon)
            texts.append(data_ann['attributes']['translation'])
    return texts, polygons


def get_pred_text_for_gt_polygon(gt_polygon, pred_data):
    max_iou = 0
    pred_text_for_gt_bbox = ''
    matching_idx = None
    gt_polygon = contour2shapely(gt_polygon)
    for idx, prediction in enumerate(pred_data['predictions']):
        if prediction.get('matched') is None:
            shapely_polygon = prediction['shapely_polygon']
            pred_text = prediction['text']
            iou = iou_polygon(gt_polygon, shapely_polygon)

            if iou > max_iou:
                max_iou = iou
                pred_text_for_gt_bbox = pred_text
                matching_idx = idx

    # to prevent matching one predicted bbox to several ground true bboxes
    if matching_idx is not None:
        pred_data['predictions'][matching_idx]['matched'] = True
    return pred_text_for_gt_bbox


def get_pred_data(img_name, pred_jsons_dir, pred_class_names):
    pred_json_name = os.path.splitext(img_name)[0] + '.json'
    pred_json_path = os.path.join(pred_jsons_dir, pred_json_name)
    with open(pred_json_path, 'r') as f:
        all_pred_data = json.load(f)

    # get prediction of only certain classes
    pred_data = {}
    pred_data["predictions"] = []
    for prediction in all_pred_data["predictions"]:
        if prediction['class_name'] in pred_class_names:
            pred_data["predictions"].append(prediction)

    # convert contour to shapley polygon
    for prediction in pred_data["predictions"]:
        polygon = prediction["polygon"]
        prediction["shapely_polygon"] = contour2shapely(polygon)
    return pred_data


def evaluate_pipeline(
    annotation_json_path, ann_class_names, pred_class_names, pred_jsons_dir,
):
    """
    Calculate the evaluation metric for the pipeline by matching the
    predicted segmentation and the ground truth bboxes (or polygons) and thus
    comparing the predicted OCR texts and the gt texts.

    Args:
        ann_class_names (list of inst): A list of class names from
            annotation.json to evaluate.
        pred_class_names (list of str): A list of class names from prediction
            to evaluate.
        pred_jsons_dir (str): Path to folder with predicted json for each image.
            Each json should have name of the image from annotation_json_path
            and have format like this:
            {
                "predictions": [
                    {
                        "polygon": list,  # the coordinates of the polygon [ [x1,y1], [x2,y2], ..., [xN,yN] ]
                        "text": str  # predicted text
                    },
                    ...
                ]
            }
        annotation_json_path (str): Path to the segmentation annotation json
            in COCO format. The json should have the following format:
            {
                "images": [
                    {
                        "file_name": str,  # name of the image file
                        "id": int  # image id
                    },
                    ...
                ],
                "annotations": [
                    {
                        "image_id": int,  # the index of the image on which the polygon is located
                        "category_id": int,  # the polygon’s category index
                        "attributes": {"translation": str},  # text in the polygon
                        "segmentation": list  # the coordinates of the polygon
                    },
                    ...
                ]
            }
    """
    with open(annotation_json_path, 'r') as f:
        data = json.load(f)

    acc_avg = AverageMeter()
    wer_avg = AverageMeter()
    cer_avg = AverageMeter()
    for data_img in tqdm(data['images']):
        img_name = data_img['file_name']
        image_id = data_img['id']

        texts_from_image, polygons_from_image = \
            get_data_from_image(data, image_id, ann_class_names)

        pred_data = get_pred_data(img_name, pred_jsons_dir, pred_class_names)

        # find predicted text for each ground true polygon
        pred_texts = []
        for gt_polygon in polygons_from_image:
            pred_texts.append(
                get_pred_text_for_gt_polygon(gt_polygon, pred_data)
            )

        # to penalty false positive prediction, that were not matched with gt
        for prediction in pred_data['predictions']:
            if prediction.get('matched') is None:
                pred_texts.append(prediction['text'])
                texts_from_image.append('')

        num_samples = len(pred_texts)
        acc_avg.update(get_accuracy(texts_from_image, pred_texts), num_samples)
        wer_avg.update(wer(texts_from_image, pred_texts), num_samples)
        cer_avg.update(cer(texts_from_image, pred_texts), num_samples)

    print(f'acc: {acc_avg.avg:.4f}')
    print(f'wer: {wer_avg.avg:.4f}')
    print(f'cer: {cer_avg.avg:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_json_path', type=str, required=True,
                        help='Path to json with segmentation dataset'
                        'annotation in COCO format.')
    parser.add_argument('--ann_class_names', nargs='+', type=str, required=True,
                        help='Category indexes (separated by spaces) from '
                        'annotation.json for evaluation.')
    parser.add_argument('--pred_class_names', nargs='+', type=str, required=True,
                        help='Class name from prediciton json for evaluation.')
    parser.add_argument('--pred_jsons_dir', type=str, required=True,
                        help='Path to folder with predicted json for each image.')
    args = parser.parse_args()

    evaluate_pipeline(args.annotation_json_path, args.ann_class_names,
                      args.pred_class_names, args.pred_jsons_dir)
