# Pipeline of segmentation and OCR model

## Quick setup and start

- Nvidia drivers >= 470, CUDA >= 11.4
- [Docker](https://docs.docker.com/engine/install/ubuntu/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

The provided [Dockerfile](Dockerfile) is supplied to build an image with CUDA support and cuDNN.

Also you can install the necessary python packages via [requirements.txt](requirements.txt)

### Preparations

- Clone the repo.
- Clone the [SEGM-model](https://github.com/sberbank-ai/SEGM-model) and [OCR-model](https://github.com/sberbank-ai/OCR-model) to the OCR-pipeline folder.
- Download weights and config-files of segmentation and OCR models to the `data/` folder.
- `sudo make all` to build a docker image and create a container.
  Or `sudo make all GPUS=device=0 CPUS=10` if you want to specify gpu devices and limit CPU-resources.

## Configuring the model

You can change the [pipeline_config.json](scripts/pipeline_config.json) (or make a copy of the file). In the config you can find the postprocessing parameters.

### Class specific parameters

Parameters in the "classes"-dict are set individually for each class. The number of classes and their names must correspond to the prediction classes of the segmentation model.

```
{
    "classes": {
        "pupil_text": {
            "postprocess": {
                "upscale_bbox": [1.4, 2.3]
            }
        },
		...
    }
}
```

Postprocessing settings:

- `upscale_bbox` - Tuple of (x, y) upscale parameters of the predicted bbox to increase it and capture large areas of the image.

## Inference

An example of using the model can be found in [inference_pipeline.ipynb](scripts/inference_pipeline.ipynb).

To evaluate the accuracy of text recognition (OCR prediction when combined with segmentation model), you can use [evaluate](scripts/evaluate.py) script (you first need to generate model predictions, an example in [inference_pipeline_on_dataset.ipynb](scripts/inference_pipeline_on_dataset.ipynb)).

