import cv2
import math
import numpy as np
import pyclipper
from shapely.geometry import Polygon

from segm.predictor import SegmPredictor
from ocr.predictor import OcrPredictor

from ocrpipeline.utils import img_crop
from ocrpipeline.config import Config
from ocrpipeline.linefinder import (
    add_polygon_center, add_page_idx_for_lines, add_line_idx_for_lines,
    add_line_idx_for_words, add_column_idx_for_words, add_word_indexes
)


def is_valid_polygon(polygon):
    """Check if a polygon is valid. Return True if valid and False otherwise.

    Args:
        polygon (shapely.geometry.Polygon): The polygon.
    """
    if (
        polygon.length < 1
        or polygon.area <= 0
    ):
        return False
    return True


def change_contour_size(polygon, scale_ratio=1):
    """
    Args:
        polygon (np.array): Array of polygon coordinates
            np.array([[x, y], ...])
        scale_ratio (Float): The scale scale_ratio to change the contour.
    """
    poly = Polygon(polygon)
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(polygon, pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON)
    if not is_valid_polygon(poly):
        return None
    distance = int(poly.area * (1 - scale_ratio ** 2) / poly.length)
    scaled_polygons = pco.Execute(-distance)
    return scaled_polygons


def get_upscaled_bbox(bbox, upscale_x=1, upscale_y=1):
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
    """Get bbox from contour."""
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
    """Define angle between two vectors. Outpur angle always positive."""
    vector_1 = [x1, y1]
    vector_2 = [x2, y2]
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    radian = np.arccos(dot_product)
    return math.degrees(radian)


def get_angle_by_fitline(contour):
    """Get angle of contour using cv2.fitLine."""
    vx, vy, x, y = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = get_angle_between_vectors(vx[0], vy[0])
    # get_line_angle return angle between vectors always positive
    # so multiply by minus one if the line is negative
    if vy > 0:
        angle *= -1
    return angle


def get_angle_by_minarearect(contour):
    """Get angle of contour using cv2.minAreaRect."""
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


class SegmPrediction:
    def __init__(self, pipeline_config, model_path, config_path, device):
        self.segm_predictor = SegmPredictor(
            model_path=model_path,
            config_path=config_path,
            device=device
        )

    def __call__(self, image, pred_img):
        pred_img = self.segm_predictor(image)
        return image, pred_img


class PrepareJSON:
    """Prepare final json to saving on disk. Remove unused subdicts.
    """
    def __init__(self, pipeline_config):
        self.elements_to_remove = ['crop', 'polygon_center']

    def __call__(self, image, pred_img):
        for prediction in pred_img['predictions']:
            for element in self.elements_to_remove:
                if element in prediction:
                    del prediction[element]
        return image, pred_img


class OCRPrediction:
    def __init__(
        self, pipeline_config, model_path, config_path, lm_path,
        classes_to_ocr, device
    ):
        self.classes_to_ocr = classes_to_ocr
        self.ocr_predictor = OcrPredictor(
            model_path=model_path,
            config_path=config_path,
            lm_path=lm_path,
            device=device,
        )

    def __call__(self, image, pred_img):
        crops = []
        indexes = []
        for idx, prediction in enumerate(pred_img['predictions']):
            if prediction['class_name'] in self.classes_to_ocr:
                crops.append(prediction['crop'])
                indexes.append(idx)
        text_preds = self.ocr_predictor(crops)
        for idx, text_pred in zip(indexes, text_preds):
            pred_img['predictions'][idx]['text'] = text_pred
        return image, pred_img


class LineFinder:
    """Heuristic methods to define indexes of rows, columns and pages for
    polygons on the image.

    Args:
        line_classes (list of strs): List of line class names.
        text_classes (list of strs): List of text class names.
    """

    def __init__(self, pipeline_config, line_classes, text_classes):
        self.line_classes = line_classes
        self.text_classes = text_classes

    def __call__(self, image, pred_img):
        _, img_w = image.shape[:2]
        add_polygon_center(pred_img)
        add_page_idx_for_lines(pred_img, self.line_classes, img_w, .25)
        add_line_idx_for_lines(pred_img, self.line_classes)
        add_line_idx_for_words(pred_img, self.line_classes, self.text_classes)
        add_column_idx_for_words(pred_img, self.text_classes)
        add_word_indexes(pred_img, self.text_classes)
        return image, pred_img


class RestoreImageAngle:
    """Define the angle of the image and rotates the image and contours to
    this angle.

    Args:
        pipeline_config (ocrpipeline.config.Config): The pipeline config.json.
        restoring_class_names (list of str):  List of class names using find
            angle of the image.
        min_angle_to_rotate (int): The safe range of angles within which image
            rotation does not occur (-min_angle_to_rotate; min_angle_to_rotate)
    """

    def __init__(
        self, pipeline_config, restoring_class_names, min_angle_to_rotate=0.5
    ):
        self.restoring_class_names = restoring_class_names
        self.min_angle_to_rotate = min_angle_to_rotate

    def __call__(self, image, pred_img):
        contours = []
        restoring_contours = []
        for prediction in pred_img['predictions']:
            contour = prediction['polygon']
            contour = np.array([contour])
            contours.append(contour)
            if prediction['class_name'] in self.restoring_class_names:
                restoring_contours.append(contour)

        angle = get_image_angle(restoring_contours)
        if abs(angle) > self.min_angle_to_rotate:
            image, contours = rotate_image_and_contours(image, contours, -angle)

        for prediction, contour in zip(pred_img['predictions'], contours):
            contour = [[int(i[0]), int(i[1])] for i in contour[0]]
            prediction['rotated_polygon'] = contour
        return image, pred_img


class BboxFromContour:
    def __call__(self, image, crop, bbox, contour):
        bbox = contour2bbox(np.array([contour]))
        return crop, bbox, contour


class CropByBbox:
    def __call__(self, image, crop, bbox, contour):
        crop = img_crop(image, bbox)
        return crop, bbox, contour


class MakeMaskedCrop:
    """Make crop by mask: blacken area outside predicted polygon."""
    def __call__(self, image, crop, bbox, contour):
        pts = np.array(contour)
        pts = pts - pts.min(axis=0)
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        return crop, bbox, contour


class UpscaleContour:
    """Upscale contour as it could have been shrinked during segmentation
    training.
    """
    def __init__(self, upscale_contour):
        self.upscale_contour = upscale_contour

    def __call__(self, image, crop, bbox, contour):
        upscaled_contour = \
            change_contour_size(contour, self.upscale_contour)
        if upscaled_contour is None:
            return crop, bbox, contour
        # take zero contour (when upscaling only one contour could be returned)
        upscaled_contour = upscaled_contour[0]
        # coords shouldn't be outside image after upscaling
        upscaled_contour = [[max(0, i[0]), max(0, i[1])]
                            for i in upscaled_contour]
        return crop, bbox, upscaled_contour


class UpscaleBbox:
    def __init__(self, upscale_bbox):
        self.upscale_bbox = upscale_bbox

    def __call__(self, image, crop, bbox, contour):
        bbox = get_upscaled_bbox(
            bbox=bbox,
            upscale_x=self.upscale_bbox[0],
            upscale_y=self.upscale_bbox[1]
        )
        return crop, bbox, contour


class RotateVerticalCrops:
    """Rotate vertical text crops (e.g. text in the margins).

    Args:
        h2w_ratio (float): height to width ratio to consider text as vertival.
        clockwise (bool): Rotate crop clockwise or counterclockwise.
    """

    def __init__(self, h2w_ratio=5, clockwise=False):
        self.h2w_ratio = h2w_ratio
        self.flip_code = 0
        if clockwise:
            self.flip_code = 1

    def __call__(self, image, crop, bbox, contour):
        h, w = crop.shape[:2]
        if h / w >= self.h2w_ratio:
            rotated_crop = cv2.transpose(crop)
            rotated_crop = cv2.flip(rotated_crop, flipCode=self.flip_code)
            return rotated_crop, bbox, contour
        return crop, bbox, contour


CONTOUR_PROCESS_DICT = {
    "BboxFromContour": BboxFromContour,
    "UpscaleBbox": UpscaleBbox,
    "CropByBbox": CropByBbox,
    "RotateVerticalCrops": RotateVerticalCrops,
    "UpscaleContour": UpscaleContour,
    "MakeMaskedCrop": MakeMaskedCrop
}


class ClassContourPosptrocess:
    """Class to handle postprocess functions for bboxs and contours."""

    def __init__(self, pipeline_config):
        self.class2process_funcs = {}
        for class_name, params in pipeline_config.get_classes().items():
            self.class2process_funcs[class_name] = []
            for process_name, args in params['contour_posptrocess'].items():
                self.class2process_funcs[class_name].append(
                    CONTOUR_PROCESS_DICT[process_name](**args)
                )

    def __call__(self, image, pred_img):
        for prediction in pred_img['predictions']:
            if prediction['class_name'] in self.class2process_funcs:
                process_funcs = \
                    self.class2process_funcs[prediction['class_name']]
                bbox = None
                crop = None
                contour = prediction['rotated_polygon']
                for process_func in process_funcs:
                    crop, bbox, contour = \
                        process_func(image, crop, bbox, contour)
                prediction['rotated_polygon'] = contour
                prediction['rotated_bbox'] = bbox
                prediction['crop'] = crop
        return image, pred_img


class ImageToBGR:
    def __init__(self, pipeline_config, input_format="BGR"):
        self.input_format = input_format

    def __call__(self, image, pred_img):
        if self.input_format == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, pred_img


MAIN_PROCESS_DICT = {
    "ImageToBGR": ImageToBGR,
    "SegmPrediction": SegmPrediction,
    "ClassContourPosptrocess": ClassContourPosptrocess,
    "RestoreImageAngle": RestoreImageAngle,
    "OCRPrediction": OCRPrediction,
    "PrepareJSON": PrepareJSON,
    "LineFinder": LineFinder
}


class PipelinePredictor:
    """Main class to handle sub-classes which make preiction pipeline loop:
    from segmentatino to ocr models. All pipeline sub-classes should be
    listed in pipeline_config.json in main_process-dict.

    Args:
        pipeline_config_path (str): A path to the pipeline config.json.
    """

    def __init__(self, pipeline_config_path):
        self.config = Config(pipeline_config_path)
        self.main_process_funcs = []
        for process_name, args in self.config.get('main_process').items():
            self.main_process_funcs.append(
                MAIN_PROCESS_DICT[process_name](
                    pipeline_config=self.config,
                    **args)
            )

    def __call__(self, image):
        """
        Args:
            image (np.array): An input image in BGR format.

        Returns:
            rotated_image (np.array): The input image which was rotated to
                restore rotation angle.
            pred_data (dict): A result dict for the input image.
                {
                    'image': {'height': Int, 'width': Int} params of the input image,
                    'predictions': [
                        {
                            'polygon': [ [x1,y1], [x2,y2], ..., [xN,yN] ] initial polygon
                            'bbox': [x_min, y_min, x_max, y_max] initial bounding box
                            'class_name': str, class name of the polygon.
                            'text': predicted text.
                            'rotated_bbox': [x_min, y_min, x_max, y_max] processed
                                bbox for a rotated image with the restored angle
                            'rotated_polygon': [ [x1,y1], [x2,y2], ..., [xN,yN] ]
                                processed polygon for a rotated image with the restored angle
                            'page_idx': int, The page index of the polygon.
                            'line_idx': int, The line index of the polygon within given page.
                            'column_idx': int, The column index of the polygon within
                                given line and page. The value exists only for text
                                classes that intersect the lines.
                            'word_idx': int, The order index of the word polygon within
                                given line and page. The value exists only for text
                                classes that intersect the lines.
                        },
                        ...
                    ]

                }
        """
        pred_img = None
        for process_func in self.main_process_funcs:
            image, pred_img = process_func(image, pred_img)
        return image, pred_img
