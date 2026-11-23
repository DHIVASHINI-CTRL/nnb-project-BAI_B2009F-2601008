"""Predict fresh/rotten for one new image.  python src/predict.py --image path.jpg"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from data import load_image_tensor
from model import FruitCNN

ap = argparse.ArgumentParser()
ap.add_argument("--image", required=True)
ap.add_argument("--name", default="final")
ap.add_argument("--model-dir", default="models")
args = ap.parse_args()

mdir = Path(args.model_dir)
cfg = json.loads((mdir / args.name / "history.json").read_text())["config"]
model = FruitCNN(cfg["dropout"])
model.load_state_dict(torch.load(mdir / args.name / "model.pt", map_location="cpu"))
model.eval()

x = load_image_tensor(Image.open(args.image)).unsqueeze(0)
with torch.no_grad():
    p_rotten = torch.sigmoid(model(x)).item()
label = "rotten" if p_rotten > 0.5 else "fresh"
print(f"Prediction: {label}  (P(rotten) = {p_rotten:.3f})")
