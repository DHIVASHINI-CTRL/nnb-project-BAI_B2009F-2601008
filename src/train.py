"""Training script. Examples:
  python src/train.py                               # baseline
  python src/train.py --name exp1 --dropout 0.0     # experiment (change ONE thing)
  python src/train.py --name final --final          # best config; evaluates test set ONCE
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import FruitDataset, make_split, seed_everything, SEED
from model import FruitCNN

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="baseline")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--dropout", type=float, default=0.3)
ap.add_argument("--weight-decay", type=float, default=0.0)
ap.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
ap.add_argument("--augment", type=int, default=0)
ap.add_argument("--fruits", default="", help="e.g. apple,banana (empty = all)")
ap.add_argument("--final", action="store_true", help="evaluate test set once")
ap.add_argument("--data-dir", default="data")
ap.add_argument("--model-dir", default="models")
args = ap.parse_args()

seed_everything(SEED)
fruits = [f.strip().lower() for f in args.fruits.split(",") if f.strip()] or None
out = Path(args.model_dir) / args.name
out.mkdir(parents=True, exist_ok=True)

split = make_split(args.data_dir, Path(args.model_dir) / "split.json", fruits)
print({k: len(split[k]) for k in ("train", "val", "test")}, "duplicates removed:", split["duplicates_removed"])

augment = None
if args.augment:
    from torchvision import transforms
    augment = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)])

g = torch.Generator().manual_seed(SEED)
train_dl = DataLoader(FruitDataset(args.data_dir, split["train"], augment), batch_size=args.batch,
                      shuffle=True, generator=g, num_workers=0)
val_dl = DataLoader(FruitDataset(args.data_dir, split["val"]), batch_size=64, num_workers=0)

model = FruitCNN(args.dropout)
loss_fn = nn.BCEWithLogitsLoss()
if args.optimizer == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)


def run_epoch(dl, train):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in dl:
            logits = model(x)
            loss = loss_fn(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(y)
            correct += ((logits > 0).float() == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val, best_epoch = float("inf"), 0
for ep in range(1, args.epochs + 1):
    tl, ta = run_epoch(train_dl, True)
    vl, va = run_epoch(val_dl, False)
    for k, v in zip(hist, (tl, vl, ta, va)):
        hist[k].append(v)
    print(f"epoch {ep:2d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
    if vl < best_val:  # keep the best epoch by validation loss (a simple early-stopping style checkpoint)
        best_val, best_epoch = vl, ep
        torch.save(model.state_dict(), out / "model.pt")

summary = {"config": vars(args), "best_epoch": best_epoch, "best_val_loss": best_val,
           "best_val_acc": hist["val_acc"][best_epoch - 1], "history": hist}
(out / "history.json").write_text(json.dumps(summary, indent=2))
print(f"Best epoch {best_epoch}: val loss {best_val:.4f}, val acc {summary['best_val_acc']:.4f}")

# Loss / accuracy curves
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(hist["train_loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
ax[1].plot(hist["train_acc"], label="train"); ax[1].plot(hist["val_acc"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
for a in ax: a.set_xlabel("epoch")
plt.suptitle(args.name); plt.tight_layout(); plt.savefig(out / "curves.png", dpi=120); plt.close()

if args.final:
    # TEST SET USED ONCE, with the best checkpoint
    import numpy as np
    from PIL import Image
    model.load_state_dict(torch.load(out / "model.pt"))
    model.eval()
    test_ds = FruitDataset(args.data_dir, split["test"])
    preds, ys = [], []
    with torch.no_grad():
        for i in range(0, len(test_ds), 64):
            batch = [test_ds[j] for j in range(i, min(i + 64, len(test_ds)))]
            x = torch.stack([b[0] for b in batch]); y = torch.stack([b[1] for b in batch])
            preds += (model(x) > 0).int().tolist(); ys += y.int().tolist()
    preds, ys = np.array(preds), np.array(ys)
    tp = int(((preds == 1) & (ys == 1)).sum()); fp = int(((preds == 1) & (ys == 0)).sum())
    fn = int(((preds == 0) & (ys == 1)).sum()); tn = int(((preds == 0) & (ys == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    metrics = {"accuracy": (tp + tn) / len(ys), "precision": prec, "recall": rec,
               "f1": 2 * prec * rec / max(prec + rec, 1e-9), "confusion_matrix": [[tn, fp], [fn, tp]],
               "majority_baseline": max((ys == 0).mean(), (ys == 1).mean()), "n_test": len(ys)}
    (out / "train_test_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("FINAL TEST METRICS (reported in training):", json.dumps(metrics, indent=2))

    # Confusion matrix figure
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow([[tn, fp], [fn, tp]], cmap="Blues")
    for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
        ax.text(j, i, v, ha="center", va="center")
    ax.set_xticks([0, 1], ["fresh", "rotten"]); ax.set_yticks([0, 1], ["fresh", "rotten"])
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=120); plt.close()

    # Wrong predictions (first 10) for error analysis
    wrong = [i for i in range(len(ys)) if preds[i] != ys[i]][:10]
    if wrong:
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        for a in axes.ravel(): a.axis("off")
        for a, i in zip(axes.ravel(), wrong):
            path, label = split["test"][i]
            a.imshow(Image.open(Path(args.data_dir) / path).convert("RGB"))
            a.set_title(f"true: {['fresh','rotten'][label]}\npred: {['fresh','rotten'][preds[i]]}", fontsize=9)
        plt.tight_layout(); plt.savefig(out / "wrong_predictions.png", dpi=120); plt.close()
        (out / "wrong_predictions.txt").write_text("\n".join(split["test"][i][0] for i in range(len(ys)) if preds[i] != ys[i]))
    print(f"Saved to {out}. Copy {out} and models/split.json to the test machine.")
