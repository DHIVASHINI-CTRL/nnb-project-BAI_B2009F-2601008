# Fresh vs Rotten Fruit Classification (CNN, PyTorch, Docker)

BAI13143 Neural Network Basics - individual project.
Binary image classification: fresh (0) vs rotten (1). Output: 1 logit + BCEWithLogitsLoss.

## Dataset
Kaggle "Fruits fresh and rotten for classification" (record exact name, URL and licence here).
Unzip so that folders like `freshapples`, `rottenapples`, `freshbanana`... are somewhere under `data/`.
Images are merged, de-duplicated by MD5, then split 70/15/15 (seed 42, stratified). The dataset's own train/test folders are ignored.

## Run training (container 1)
    docker build -f Dockerfile.train -t nn-train .
    docker run --rm -v ./data:/app/data -v ./models:/app/models nn-train                      # baseline
    docker run --rm -v ./data:/app/data -v ./models:/app/models nn-train --name exp1 --dropout 0.0
    docker run --rm -v ./data:/app/data -v ./models:/app/models nn-train --name exp2 --augment 1
    docker run --rm -v ./data:/app/data -v ./models:/app/models nn-train --name final --final  # best config + test set ONCE
Add the same `--fruits apple,banana` flag to every run if you restrict the fruits (first run creates `models/split.json`).

## Run testing (container 2, different machine)
Copy `models/` (with `split.json` and `final/`) and `data/` to the other machine, then:
    docker build -f Dockerfile.test -t nn-test .
    docker run --rm -v ./data:/app/data -v ./models:/app/models nn-test
    docker run --rm -v ./models:/app/models -v ./samples:/app/samples nn-test src/predict.py --image /app/samples/apple1.jpg

## Reproducibility
Pinned versions in requirements-*.txt, seed 42 for random/numpy/torch/DataLoader, CPU only.
`test.py` compares its metrics with `models/final/train_test_metrics.json` and prints MATCH.

## GenAI declaration
Describe honestly how you used Claude (e.g. help with project structure and debugging).
