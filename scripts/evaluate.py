import argparse
import os
import json
from tqdm import tqdm
import numpy as np

from ocrpipeline.utils import AverageMeter
from ocrpipeline.metrics import get_accuracy, cer, wer, iou_bbox, iou_polygon


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


def get_data_from_image(data, image_id, category_ids):
    texts = []
    polygons = []
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


def get_pred_text_for_gt_polygon(gt_polygon, pred_data, evaluate_by_bbox):
    max_iou = 0
    pred_text_for_gt_bbox = ''
    matching_idx = None
    gt_bbox = polygon2bbox(gt_polygon)
    for idx, prediction in enumerate(pred_data['predictions']):
        if prediction.get('matched') is None:
            polygon = prediction['polygon']
            pred_text = prediction['text']
            if evaluate_by_bbox:
                bbox = polygon2bbox(polygon)
                iou = iou_bbox(gt_bbox, bbox)
            else:
                iou = iou_polygon(gt_polygon, polygon)

            if iou > max_iou:
                max_iou = iou
                pred_text_for_gt_bbox = pred_text
                matching_idx = idx

    # to prevent matching one predicted bbox to several ground true bboxes
    if matching_idx is not None:
        pred_data['predictions'][matching_idx]['matched'] = True
    return pred_text_for_gt_bbox


def main(args):
    with open(args.annotation_json_path, 'r') as f:
        data = json.load(f)

    acc_avg = AverageMeter()
    wer_avg = AverageMeter()
    cer_avg = AverageMeter()
    for data_img in tqdm(data['images']):
        img_name = data_img['file_name']
        image_id = data_img['id']

        texts_from_image, polygons_from_image = \
            get_data_from_image(data, image_id, args.category_ids)

        pred_json_name = os.path.splitext(img_name)[0] + '.json'
        pred_json_path = os.path.join(args.pred_jsons_dir, pred_json_name)
        with open(pred_json_path, 'r') as f:
            pred_data = json.load(f)

        # find predicted text for each ground true polygon
        pred_texts = []
        for gt_polygon in polygons_from_image:
            pred_texts.append(
                get_pred_text_for_gt_polygon(
                    gt_polygon, pred_data, args.evaluate_by_bbox)
            )

        # to penalty false positive prediction, that were not matched with gt
        for prediction in pred_data['predictions']:
            if (
                prediction.get('matched') is None
                and prediction['text']  # ignore empty prediction
            ):
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
                        'annotation in COCO format')
    parser.add_argument('--category_ids', nargs='+', type=int, required=True,
                        help='Category indexes (separated by spaces) from '
                        'annotation_json_path for evaluation.')
    parser.add_argument('--pred_jsons_dir', type=str, required=True,
                        help='Path to folder with predicted json for each image.')
    parser.add_argument('--evaluate_by_bbox', action='store_true',
                        help='To evaluate prediction by bbox.')
    args = parser.parse_args()

    main(args)
