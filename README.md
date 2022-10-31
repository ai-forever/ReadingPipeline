# Reading Pipeline

This is a pipeline for text reading. It combines the [OCR](https://github.com/ai-forever/OCR-model) and [Segmentation](https://github.com/ai-forever/SEGM-model) models into a single pipeline and allows to segment an input image, than crop text regions from it and, finally, read these texts using OCR.


## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

### Models

[Weights for reading manuscripts of Peter the Great](https://huggingface.co/sberbank-ai/ReadingPipeline-Peter), and [Peter dataset](https://huggingface.co/datasets/sberbank-ai/Peter)

[Weights for reading school notebooks](https://huggingface.co/sberbank-ai/ReadingPipeline-notebooks), and [notebooks dataset](https://huggingface.co/datasets/sberbank-ai/school_notebooks_RU)

### Demo

[There](https://huggingface.co/spaces/sberbank-ai/PeterRecognition) you can find demo of ReadingPipeline model on Peter the Great handwritings.

### Preparations

- Clone the repo.
- Download weights and config-files of segmentation and OCR models to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
   Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

If you don't want to use Docker, you can install dependencies via requirements.txt

## Configuring the model

You can change parameters of the pipeline in the [pipeline_config.json](scripts/pipeline_config.json) (or make a copy of the file).

### Main pipeline loop

The main_process-dict defines the order of the main processing methods and models that make up the pipeline loop. Classes are initialized with the parameters specified in the config, and are called one after the other in the order that is defined in the config.

PipelinePredictor - the class responsible for assembling the pipeline, and is located in [ocrpipeline/predictor.py](ocrpipeline/predictor.py). To add a new class to the pipeline, you need to add it to the `MAIN_PROCESS_DICT` dictionary in [ocrpipeline/predictor.py](ocrpipeline/predictor.py) and also specify it in the main_process-dict in the config at the point in the chain in which the class should be called.

```
"main_process": {
    "SegmPrediction": {...},
    "RestoreImageAngle": {...},
    "ClassContourPosptrocess": {...},
    "OCRPrediction": {...},
}
```

### Class specific parameters

Parameters in the classes-dict are set individually for each class. The names of the classes must correspond to the prediction class names of the segmentation model.

The contour_posprocess-dict defines the order of the contour processing, predicted by the segmentation model. Classes are initialized with the parameters specified in the config, and are called one after the other in the order that is defined in the config.

ClassContourPosptrocess is the class responsible for assembling and calling contour_posptrocess methods, and is located in [ocrpipeline/predictor.py](ocrpipeline/predictor.py). To add a new class to the pipeline, you need to add it to the `CONTOUR_PROCESS_DICT` dictionary in [ocrpipeline/predictor.py](ocrpipeline/predictor.py) and also specify it in the contour_posprocess-dict in the config at the point in the chain in which the class should be called.

```
"classes": {
    "shrinked_pupil_text": {
        "contour_posptrocess": {
            "BboxFromContour": {},
            "UpscaleBbox": {"upscale_bbox": [1.4, 2.3]}
        }
    },
	...
}
```

## Inference

An example of using the model can be found in [inference_pipeline.ipynb](scripts/inference_pipeline.ipynb).

To evaluate the accuracy of text recognition (OCR prediction when combined with segmentation model), you can use [evaluate](scripts/evaluate.py) script (you first need to generate model predictions, an example in [inference_pipeline_on_dataset.ipynb](scripts/inference_pipeline_on_dataset.ipynb)).

