# Reading Pipeline

This is a pipeline for text detection and reading. It combines the [OCR](https://github.com/ai-forever/OCR-model) and [Segmentation](https://github.com/ai-forever/SEGM-model) models into the single pipeline and allows to segment an input image, then crop text regions from it and, finally, read these texts using OCR.

## Demo

A [web demo](https://huggingface.co/spaces/sberbank-ai/PeterRecognition) (on hugging face) of ReadingPipeline for the Peter the Great dataset.

Also there is a [demo-ReadPipeline.ipynb](scripts/demo-ReadPipeline.ipynb) with demo usage of ReadingPipeline (you can run it in your Google Colab).

### Models

[Weights for reading manuscripts of Peter the Great](https://huggingface.co/sberbank-ai/ReadingPipeline-Peter), and [Peter dataset](https://huggingface.co/datasets/sberbank-ai/Peter)

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

## Preparations

- Clone the repo.
- Download weights and config-files of segmentation and OCR models to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
   Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

If you don't want to use Docker, you can install dependencies via requirements.txt

## Configuring the pipeline

You can change parameters of the pipeline in the [pipeline_config.json](scripts/pipeline_config.json).

### Main pipeline loop

The `main_process`-dict defines the order of the main processing methods that make up the pipeline loop. Classes are initialized with the parameters specified in the config, and are called one after the other in the predefined order.

PipelinePredictor - the class responsible for assembling the pipeline, and is located in [ocrpipeline/predictor.py](ocrpipeline/predictor.py). To add a new class to the pipeline, you need to add it to the `MAIN_PROCESS_DICT` dictionary in [ocrpipeline/predictor.py](ocrpipeline/predictor.py) and also specify it in the `main_process`-dict in the config at the point in the chain from which the class should be called.

```
"main_process": {
    "SegmPrediction": {...},
    "RestoreImageAngle": {...},
    "ClassContourPosptrocess": {...},
    "OCRPrediction": {...},
    "LineFinder": {...},
    ...
}
```

### Models runtime, ONNX

You can specify runtime method for OCR and segmentation models.

```
"main_process": {
    "SegmPrediction": {
        "model_path": "/path/to/model.ckpt",
        "config_path": "/path/to/config.json",
        "num_threads": 8,
        "device": "cuda",
        "runtime": "Pytorch"  # here you can chose runtime method
    },
    ...
}
```

You can chose runtime method from several options: "Pytorch" (cuda and cpu devices), "ONNX" (only cpu is allowed) or "OpenVino" (only cpu).

### Class specific parameters

Parameters in the `classes`-dict are set individually for each class. The names of the classes must correspond to the class names of the segmentation model.

The `contour_posprocess`-dict defines the order of the contour processing, predicted by the segmentation model. Classes are initialized with the parameters specified in the config, and are called one after the other in the predefined order.

`ClassContourPosptrocess` is the class responsible for assembling and calling `contour_posptrocess` methods, and is located in [ocrpipeline/predictor.py](ocrpipeline/predictor.py). To add a new class to the pipeline, you need to add it to the `CONTOUR_PROCESS_DICT` dictionary in [ocrpipeline/predictor.py](ocrpipeline/predictor.py) and also specify it in the `contour_posprocess`-dict in the config at the point in the chain from which the class should be called.

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

An example of model inference can be found in [inference_pipeline.ipynb](scripts/inference_pipeline.ipynb).

To evaluate the pipeline accuracy (the OCR-model combined with the SEGM-model), you can use [evaluate](scripts/evaluate.py) script (you first need to generate model predictions, an example in [inference_pipeline_on_dataset.ipynb](scripts/inference_pipeline_on_dataset.ipynb)).

