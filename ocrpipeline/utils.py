from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np


def img_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def get_image_visualization(
    img, pred_data, classes_to_draw, fontpath,
    polygon_name='polygon', text_name='text',
    font_koef=50
):
    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h/font_koef))
    empty_img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data['predictions']:
        if prediction['class_name'] in classes_to_draw:
            contour = prediction[polygon_name]
            pred_text = prediction[text_name]
            cv2.drawContours(img, np.array([contour]), -1, (0, 255, 0), 2)
            draw.text(min(contour), pred_text, fill=0, font=font)

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
