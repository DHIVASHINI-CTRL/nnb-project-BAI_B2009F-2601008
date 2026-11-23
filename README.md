# Fresh vs Rotten Fruit Classification (CNN, PyTorch, Docker)

BAI13143 Neural Network Basics - Individual Project
Student: Dhivashini (BAI_B2009F-2601008)

Binary image classification: **fresh (0)** vs **rotten (1)** for apples, bananas and oranges.
The model is a small CNN written from scratch in PyTorch. Training and testing run in two
separate Docker containers.

## Dataset
- Name / source: Fresh and Rotten Classification by Swoyam Siddharth Nayak, Kaggle, https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification
- Licence: Community Data License Agreement - Permissive - Version 1.0
- Used folders: apples, bananas, oranges (fresh and rotten). Vegetable folders are ignored (`--fruits apple,banana,orange`).
- 13,599 images after removing 4,896 exact duplicate files.
- Split (seed 42): train 9,663 / validation 1,940 / test 1,996.
- The dataset contains edited copies of original photos (rotated, flipped, shifted, noisy). The split is **grouped by
  original photo**, so all copies of one photo stay in the same split and cannot leak from train into test.
- The dataset's own Train/Test folders are merged and re-split.
- The dataset is **not** in this repository. Download it and unzip it so that folders such as
  `data/Train/freshapples/...` and `data/Test/rottenoranges/...` exist.

## Network (548,129 trainable parameters)
| Layer | Output | Parameters |
|---|---|---|
| Input | 3x64x64 | 0 |
| Conv 3->16, 3x3, ReLU, MaxPool | 16x32x32 | 448 |
| Conv 16->32, 3x3, ReLU, MaxPool | 32x16x16 | 4,640 |
| Conv 32->64, 3x3, ReLU, MaxPool | 64x8x8 | 18,496 |
| Flatten, Dropout | 4096 | 0 |
| Dense 4096->128, ReLU | 128 | 524,416 |
| Dense 128->1 (sigmoid inside loss) | 1 | 129 |

Loss: BCEWithLogitsLoss. Optimiser: Adam (lr 0.001, batch 32). The best epoch is saved by validation loss.

## Experiments (validation set)
| Exp | What changed | Epochs | Dropout | Best val loss | Best val acc |
|---|---|---|---|---|---|
| 0 | Baseline | 15 | 0.3 | 0.0590 (epoch 14) | 0.9747 |
| 1 | More epochs | 30 | 0.3 | 0.0590 (epoch 14) | 0.9747 |
| 2 | No dropout | 30 | 0.0 | 0.0533 (epoch 15) | 0.9773 |

Final model: dropout 0.0, checkpoint at epoch 15 (lowest validation loss).

## Final test result (test set used once)
Accuracy 0.9374, precision 0.9336, recall 0.9533, F1 0.9433 (rotten = positive class).
Majority-class baseline: 0.5466. Confusion matrix [[831, 74], [51, 1040]].
The target of 95% accuracy/F1 was not reached (about 1 to 1.5 points short).

## Project structure
```
data/                  dataset (mounted, not copied into images, not in Git)
models/final/          model.pt, history.json, train_test_metrics.json, figures
models/split.json      fixed train/val/test split (needed by the test container)
models/baseline, exp1, exp2   experiment results
src/                   data.py model.py train.py test.py predict.py
Dockerfile.train       training image
Dockerfile.test        lean inference image (torch, numpy, pillow only)
requirements-*.txt     pinned versions
```

## How to run

### 1. Training container (PowerShell, inside this folder)
```
docker build -f Dockerfile.train -t nn-train .
docker run --rm -e PYTHONUNBUFFERED=1 -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" nn-train --fruits apple,banana,orange --name final --epochs 15 --dropout 0.0 --final
```
The first run builds an image cache and is slow. Other runs use `--name exp1 --epochs 30`, `--name exp2 --epochs 30 --dropout 0.0`, etc.
(On Linux/macOS use `$(pwd)` instead of `${PWD}`.)

### 2. Testing container (can run on a different machine)
Copy `models/final/` and `models/split.json` (and the test images listed in `split.json`) to the other machine, then:
```
docker build -f Dockerfile.test -t nn-test .
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" nn-test
```
Or use the published image, with no build:
```
docker pull dhivashini/nn-fruit-test:1.0
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/models:/app/models" dhivashini/nn-fruit-test:1.0
```
It prints the metrics and `MATCH: True` when they equal those reported in training.

### 3. Predict one new photo
```
docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/samples:/app/samples" nn-test src/predict.py --image /app/samples/photo.jpg
```

## Reproducibility
Library versions are pinned, seeds are fixed (42), and the CPU is used. The test container was run on a second laptop
with a different CPU (AMD Ryzen 7 170 vs Intel i5-13420H) and gave exactly the same results.

## Limitations
- The dataset contains augmented copies and some noisy labels. The split is grouped by original photo, but near-copies could remain.
- Images are reduced to 64x64, so small signs of rot (wrinkles, small spots) are often missed.
- Results are from a single run per setting.

## Generative AI declaration
I used Claude (Anthropic) as a learning and troubleshooting assistant while preparing this project. It helped me with Git setup and troubleshooting, Docker testing on a second laptop, and project documentation. I ran the model training, evaluation and Docker tests myself.

## References
- Dataset: Fresh and Rotten Classification by Swoyam Siddharth Nayak, https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification
- Paszke, A. et al. (2019) PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
- Kingma, D. and Ba, J. (2015) Adam: A Method for Stochastic Optimization. ICLR.
- Docker documentation, https://docs.docker.com
