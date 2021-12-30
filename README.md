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

## Inference

An example of using the model can be found in [inference_pipeline.ipynb](scripts/inference_pipeline.ipynb).

To evaluate the accuracy of text recognition (OCR prediction when combined with segmentation model), you can use [evaluate](scripts/evaluate.py) script (you first need to generate model predictions, an example in [inference_pipeline_on_dataset.ipynb](scripts/inference_pipeline_on_dataset.ipynb)).

