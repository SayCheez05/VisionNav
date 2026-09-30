# VisionNav

VisionNav is a student research project for learning computer vision through indoor robotics. The current work studies monocular RGB object detection on Robot@Home2; later milestones will add simulation and, after the perception baseline is better understood, low-cost stereo hardware. This repository is a research and learning project, not a deployment-ready or safety-critical robot system.

README and project status are maintained in Chinese because this is the author's primary working language.

## Project goals

- Build a reproducible object-detection workflow that can be inspected and improved, rather than a one-off demo.
- Measure generalization to unseen indoor environments with leave-one-house-out (LOHO) validation.
- Diagnose small-bottle misses and domain differences before adding more complicated training techniques.
- Later connect camera perception to a robot simulator, then assess a stereo camera within a hardware budget of about RMB 1,000.

## Current status — 2026-09-30

The complete 640-pixel candidate-v3 LOHO experiment is the current reference baseline. Across the four development homes, its unweighted means are **mAP50 0.2125**, **mAP50–95 0.1114**, and **recall 0.2571**. Bottle AP50 is **0.0891** and bottle recall is **0.1328**. These results show substantial missed detections and variation between homes; they do not support real-world deployment claims.

A controlled 960-pixel comparison was started with the same four folds and training settings. It was stopped at the user's request after three folds completed independent validation; the fourth fold, `rx2`, had reached epoch 5 and has checkpoint files, but no independent validation. The three completed folds have mean mAP50 **0.1299**, mean mAP50–95 **0.0637**, and mean recall **0.1686**. This is an incomplete comparison and must not be presented as a final four-fold result or evidence that 960 resolution is better.

The `sarmis` home was used once for a final evaluation of an earlier v2 model. It is therefore not a fresh test set and is excluded from current training, model selection, and tuning. Full progress, known issues, and unfinished work are in [Project status](docs/PROJECT_STATUS.md).

## What is included

- Image inference, dataset preparation, training, evaluation, and audit code in `src/visionnav/`.
- Conda and pip dependency definitions.
- The 640 baseline analysis and two research notes in `saycheez/`.
- Dataset and environment guidance in `data/` and `docs/`.

Datasets, personal videos and CVAT exports, model weights, generated images, run directories, machine-specific reports, and internal work logs are intentionally excluded. Obtain third-party datasets and pretrained weights from their original providers and follow their terms. The MIT license in this repository applies to VisionNav's code; it does not relicense those assets.

## Environment (Windows PowerShell)

Create the project-local Conda environment:

```powershell
conda env create --prefix .\.conda-env --file environment.yml
```

Install the CUDA-enabled PyTorch build tested for the current Windows setup, then the project dependencies:

```powershell
.\.conda-env\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.conda-env\python.exe -m pip install -r requirements.txt
.\.conda-env\python.exe -m pip install -e .
```

If the PyTorch wheel versions have changed, use the [official PyTorch installer selector](https://pytorch.org/get-started/locally/) to choose a compatible command for your operating system and NVIDIA driver.

## Run the image demo

Download the public sample image and run inference:

```powershell
New-Item -ItemType Directory -Force data, weights, outputs | Out-Null
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg" -OutFile "data\bus.jpg"
.\.conda-env\python.exe -m visionnav.infer --source data\bus.jpg --model weights\yolo26n.pt --output outputs\detection.jpg
```

The sample image and pretrained checkpoint are local assets and are not committed. The first command downloads the image; obtain the checkpoint from its original provider if it is not already available locally.

## Robot@Home2 experiment

The Robot@Home2 source data is not redistributed here. Download it from the dataset provider, review its license, and place the archives under `data/datasets/RobotAtHome2/downloads/`. The preprocessing code expects extracted source files under `data/datasets/RobotAtHome2/raw/`; generated folds are written under `data/datasets/RobotAtHome2/processed/`.

The candidate-v3 data preparation entry point is:

```powershell
.\.conda-env\python.exe -m visionnav.prepare_robotathome_cv_v3
```

After preparing the dataset and placing the compatible pretrained checkpoint at `weights/yolo26n.pt`, a fresh LOHO run can be started with a **new** output directory and summary path:

```powershell
.\.conda-env\python.exe -m visionnav.train_robotathome_cv --data-root data/datasets/RobotAtHome2/processed/yolo_cv_candidate_v3 --model weights/yolo26n.pt --project runs/robotathome2_cv_new --summary reports/robotathome2_cv_new.md --folds val_alma val_anto val_pare val_rx2 --epochs 100 --patience 15 --run-prefix candidate_new --imgsz 640 --batch 8 --device 0 --workers 0 --seed 42 --optimizer AdamW --lr0 0.001429 --warmup-bias-lr 0.0
```

This command starts a new experiment. Do not reuse existing output paths: the scripts intentionally refuse to overwrite prior runs. The paused 960 run is incomplete; its resumption method has not yet been implemented or verified.

## License and scope

VisionNav code is released under the [MIT License](LICENSE). Dataset content, pretrained checkpoints, Python packages, and sample media retain their respective licenses. See [project status](docs/PROJECT_STATUS.md) for the limitations and remaining research tasks.
