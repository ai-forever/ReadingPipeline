from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np


def img_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def get_image_visualization(img, pred_data, fontpath, font_koef=50):
    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h/font_koef))
    empty_img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data['predictions']:
        bbox = prediction['bbox']
        pred_text = prediction['text']
        cv2.rectangle(
            img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
        draw.text((bbox[0], bbox[1]), pred_text, fill=0, font=font)

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
