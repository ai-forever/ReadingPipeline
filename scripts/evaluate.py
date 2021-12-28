import cv2
import argparse
import os
import json
from tqdm import tqdm
import numpy as np

from ocrpipeline.predictor import PipelinePredictor
from ocrpipeline.utils import AverageMeter
from ocrpipeline.metrics import get_accuracy, cer, wer, iou_bbox


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[list_of_numbers[i], list_of_numbers[i+1]]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


def upscale_bbox(bbox, upscale_x=1, upscale_y=1):
    """Increase size of the bbox."""
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]

    y_change = (height * upscale_y) - height
    x_change = (width * upscale_x) - width

    x_min = max(0, bbox[0] - int(x_change/2))
    y_min = max(0, bbox[1] - int(y_change/2))
    x_max = bbox[2] + int(x_change/2)
    y_max = bbox[3] + int(y_change/2)
    return x_min, y_min, x_max, y_max


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
    bboxes = []
    for idx, data_ann in enumerate(data['annotations']):
        if (
            data_ann['image_id'] == image_id
            and data_ann['category_id'] in category_ids
            and data_ann['attributes']
            and data_ann['attributes']['translation']
            and data_ann['segmentation']
        ):
            polygon = numbers2coords(data_ann['segmentation'][0])
            bbox = polygon2bbox(polygon)
            bboxes.append(bbox)
            texts.append(data_ann['attributes']['translation'])
    return texts, bboxes


def get_pred_text_for_gt_bbox(gt_bbox, pred_data):
    max_iou = 0
    pred_text_for_gt_bbox = ''
    for prediction in pred_data['predictions']:
        bbox = prediction['bbox']
        pred_text = prediction['text']
        iou = iou_bbox(gt_bbox, bbox)
        if iou > max_iou:
            max_iou = iou
            pred_text_for_gt_bbox = pred_text
    return pred_text_for_gt_bbox


def main(args):
    predictor = PipelinePredictor(
        segm_model_path=args.segm_model_path,
        segm_config_path=args.segm_config_path,
        ocr_model_path=args.ocr_model_path,
        ocr_config_path=args.ocr_config_path,
        device=args.device
    )

    with open(args.data_json_path, 'r') as f:
        data = json.load(f)

    acc_avg = AverageMeter()
    wer_avg = AverageMeter()
    cer_avg = AverageMeter()
    for data_img in tqdm(data['images']):
        img_name = data_img['file_name']
        image_id = data_img['id']
        image = cv2.imread(os.path.join(args.image_root, img_name))

        texts_from_image, bboxes_from_image = \
            get_data_from_image(data, image_id, args.category_ids)

        pred_data = predictor(image)

        pred_texts = []
        for gt_bbox in bboxes_from_image:
            pred_texts.append(
                get_pred_text_for_gt_bbox(gt_bbox, pred_data)
            )

        num_imgs = len(pred_texts)
        acc_avg.update(get_accuracy(texts_from_image, pred_texts), num_imgs)
        wer_avg.update(wer(texts_from_image, pred_texts), num_imgs)
        cer_avg.update(cer(texts_from_image, pred_texts), num_imgs)

    print(f'acc: {acc_avg.avg:.4f}')
    print(f'wer: {wer_avg.avg:.4f}')
    print(f'cer: {cer_avg.avg:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--segm_model_path', type=str, required=True,
                        help='Path to segmentation model weights.')
    parser.add_argument('--segm_config_path', type=str, required=True,
                        help='Path to segmentation config json.')
    parser.add_argument('--ocr_model_path', type=str, required=True,
                        help='Path to OCR model weights.')
    parser.add_argument('--ocr_config_path', type=str, required=True,
                        help='Path to OCR config json.')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--data_json_path', type=str, required=True,
                        help='Path to json with segmentation dataset'
                        'annotation in COCO format')
    parser.add_argument('--image_root', type=str, required=True,
                        help='Path to folder with evaluation images')
    parser.add_argument('--category_ids', nargs='+', type=int, required=True,
                        help='Category indexes (separated by spaces) from '
                        'data_json_path for evaluation.')
    args = parser.parse_args()

    main(args)
