from segm.predictor import SegmPredictor
from ocr.predictor import OcrPredictor

from ocrpipeline.utils import img_crop


class PipelinePredictor:
    def __init__(
        self, segm_model_path, segm_config_path,
        ocr_model_path, ocr_config_path, device='cuda'
    ):
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

        for prediction in pred_img['predictions']:
            bbox = prediction['bbox']
            crop = img_crop(image, bbox)
            text_pred = self.ocr_predictor(crop)
            prediction['text'] = text_pred
        return pred_img
