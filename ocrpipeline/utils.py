from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np


def img_crop(image, bbox):
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]


def get_image_visualization(
    img, pred_data, draw_contours_classes, draw_text_classes=None,
    structured_text=None, polygon_name='polygon',
    font_koef=50
):
    h, w = img.shape[:2]
    font = ImageFont.truetype("DejaVuSans.ttf", int(h/font_koef))
    empty_img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)
    if draw_text_classes is None:
        draw_text_classes = draw_contours_classes

    for prediction in pred_data['predictions']:
        contour = prediction[polygon_name]
        pred_text = prediction.get('text', '')
        if prediction['class_name'] in draw_contours_classes:
            cv2.drawContours(img, np.array([contour]), -1, (0, 255, 0), 2)
        if (
            structured_text is None
            and prediction['class_name'] in draw_text_classes
        ):
            draw.text(min(contour), pred_text, fill=0, font=font)

    if structured_text is not None:
        row = int(h/font_koef) * 1.25
        col = int(w/font_koef) * 20
        x = row
        y = row
        for page_text in structured_text:
            for line_text in page_text:
                if line_text:
                    draw.text((x, y), ' '.join(line_text), fill=0, font=font)
                    y += row
            x += col
            y = row

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


def get_line_number_visualization(
    image, pred_img, class_names, position_name
):
    for prediction in pred_img['predictions']:
        contour = prediction['polygon']
        position_idx = prediction.get(position_name)
        if prediction['class_name'] in class_names:
            cv2.drawContours(image, np.array([contour]), -1, (0, 200, 0), 2)
            text = f'{position_idx}'
            image = cv2.putText(image, text, min(contour),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.75, (255, 20, 20), 3, cv2.LINE_AA)
    return image


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
