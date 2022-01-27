import cv2
import math
import numpy as np

from segm.predictor import SegmPredictor
from ocr.predictor import OcrPredictor

from ocrpipeline.utils import img_crop
from ocrpipeline.config import Config


def get_postprocess_bbox(contour, class_params):
    bbox = contour2bbox(contour)
    bbox = upscale_bbox(
        bbox=bbox,
        upscale_x=class_params['postprocess']['upscale_bbox'][0],
        upscale_y=class_params['postprocess']['upscale_bbox'][1]
    )
    return bbox


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


def contour2bbox(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (x, y, x + w, y + h)


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_angle_between_vectors(x1, y1, x2=1, y2=0):
    """Define angle between two vectors. Angle always positive."""
    vector_1 = [x1, y1]
    vector_2 = [x2, y2]
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radian = np.arccos(dot_product)
    return math.degrees(radian)


def get_angle_by_fitline(contour):
    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = get_angle_between_vectors(vx[0], vy[0])
    # get_line_angle return angle between vectors always positive
    # so multiply by minus one if the line is negative
    if vy > 0:
        angle *= -1
    return angle


def get_angle_by_minarearect(contour):
    rect = cv2.minAreaRect(contour)
    angle = rect[2]
    # revert angles as cv2 coordinate axis starts from up right corner
    angle *= -1
    # take the opposite angle if the rectangle is too rotated
    if angle < -45:
        angle += 90
    return angle


def rotate_image_and_contours(image, contours, angle):
    """Rotate the image and contours by the angle."""
    rotated_image, M = rotate_image(image, angle)
    rotated_contours = []
    for contour in contours:
        contour = cv2.transform(contour, M)
        rotated_contours.append(contour)
    return rotated_image, rotated_contours


def get_image_angle(contours, by_fitline=True):
    """Define the angle of the image using the contours of the words."""
    angles = []
    for contour in contours:
        if by_fitline:
            angle = get_angle_by_fitline(contour)
        else:
            angle = get_angle_by_minarearect(contour)
        angles.append(angle)
    return np.median(np.array(angles))


def rotate_image(mat, angle):
    """
    https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
    Rotates an image (angle in degrees) and expands image to avoid cropping.
    """
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat, rotation_mat


class PipelinePredictor:
    def __init__(
        self, segm_model_path, segm_config_path,
        ocr_model_path, ocr_config_path, pipeline_config_path, device='cuda'
    ):
        self.config = Config(pipeline_config_path)
        self.cls2params = self.config.get_classes()
        self.segm_predictor = SegmPredictor(
            model_path=segm_model_path,
            config_path=segm_config_path,
            device=device
        )

        self.ocr_predictor = OcrPredictor(
            model_path=ocr_model_path,
            config_path=ocr_config_path,
            device=device,
        )

    def __call__(self, image):
        pred_img = self.segm_predictor(image)

        contours = []
        class_names = []
        for prediction in pred_img['predictions']:
            contour = prediction['polygon']
            contours.append(np.array([contour]))
            class_names.append(prediction['class_name'])

        angle = get_image_angle(contours)
        image, contours = rotate_image_and_contours(image, contours, -angle)

        for idx, (contour, cls_name) in enumerate(zip(contours, class_names)):
            bbox = get_postprocess_bbox(contour, self.cls2params[cls_name])
            crop = img_crop(image, bbox)
            text_pred = self.ocr_predictor(crop)
            pred_img['predictions'][idx]['text'] = text_pred
        return pred_img
