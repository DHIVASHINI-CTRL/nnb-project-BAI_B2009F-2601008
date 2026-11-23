"""Test script: loads saved model + saved split, evaluates the test set, prints metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data import FruitDataset, seed_everything, SEED
from model import FruitCNN

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="final")
ap.add_argument("--data-dir", default="data")
ap.add_argument("--model-dir", default="models")
args = ap.parse_args()

seed_everything(SEED)
mdir = Path(args.model_dir)
split = json.loads((mdir / "split.json").read_text())
cfg = json.loads((mdir / args.name / "history.json").read_text())["config"]

model = FruitCNN(cfg["dropout"])
model.load_state_dict(torch.load(mdir / args.name / "model.pt", map_location="cpu"))
model.eval()

ds = FruitDataset(args.data_dir, split["test"])
preds, ys = [], []
with torch.no_grad():
    for i in range(0, len(ds), 64):
        batch = [ds[j] for j in range(i, min(i + 64, len(ds)))]
        x = torch.stack([b[0] for b in batch]); y = torch.stack([b[1] for b in batch])
        preds += (model(x) > 0).int().tolist(); ys += y.int().tolist()
preds, ys = np.array(preds), np.array(ys)

tp = int(((preds == 1) & (ys == 1)).sum()); fp = int(((preds == 1) & (ys == 0)).sum())
fn = int(((preds == 0) & (ys == 1)).sum()); tn = int(((preds == 0) & (ys == 0)).sum())
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
print(f"Test images : {len(ys)}")
print(f"Accuracy    : {(tp + tn) / len(ys):.4f}")
print(f"Precision   : {prec:.4f}  (rotten = positive class)")
print(f"Recall      : {rec:.4f}")
print(f"F1-score    : {f1:.4f}")
print("Confusion matrix [rows=actual fresh/rotten, cols=predicted fresh/rotten]:")
print(f"  [[{tn:4d} {fp:4d}]\n   [{fn:4d} {tp:4d}]]")
print(f"Majority-class baseline accuracy: {max((ys == 0).mean(), (ys == 1).mean()):.4f}")

ref = mdir / args.name / "train_test_metrics.json"
if ref.exists():
    r = json.loads(ref.read_text())
    same = abs(r["accuracy"] - (tp + tn) / len(ys)) < 1e-9
    print(f"Reported in training: accuracy {r['accuracy']:.4f}, f1 {r['f1']:.4f} -> MATCH: {same}")
