"""Make the 3 data-exploration figures for the report (run inside the train container).
Output goes to the evidence/ folder: fig1_counts.png, fig2_samples.png, fig3_size_brightness.png
"""
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DATA, OUT = Path("data"), Path("evidence")
OUT.mkdir(exist_ok=True)
random.seed(42)
split = json.loads(Path("models/split.json").read_text())
items = split["train"] + split["val"] + split["test"]
FRUITS = ["apple", "banana", "orange"]


def fruit(path):
    folder = Path(path).parent.name.lower()
    return next((f for f in FRUITS if f in folder), "other")


# Figure 1: images per fruit and class
fresh = [sum(1 for p, l in items if fruit(p) == f and l == 0) for f in FRUITS]
rotten = [sum(1 for p, l in items if fruit(p) == f and l == 1) for f in FRUITS]
x = np.arange(len(FRUITS))
plt.figure(figsize=(7, 4.5))
plt.bar(x - 0.2, fresh, 0.4, label="fresh", color="#4C9F70")
plt.bar(x + 0.2, rotten, 0.4, label="rotten", color="#C8553D")
for i in range(len(FRUITS)):
    plt.text(i - 0.2, fresh[i], str(fresh[i]), ha="center", va="bottom")
    plt.text(i + 0.2, rotten[i], str(rotten[i]), ha="center", va="bottom")
plt.xticks(x, FRUITS)
plt.ylabel("number of images")
plt.title("Images per fruit and class (after removing duplicates)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "fig1_counts.png", dpi=150)
plt.close()
print("Per fruit (fresh, rotten):", dict(zip(FRUITS, zip(fresh, rotten))))
print("Total:", len(items), "| fresh:", sum(fresh), "| rotten:", sum(rotten))

# Figure 2: sample photos
train = split["train"]
pick = random.sample([i for i in train if i[1] == 0], 6) + random.sample([i for i in train if i[1] == 1], 6)
fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for ax, (p, l) in zip(axes.ravel(), pick):
    ax.imshow(Image.open(DATA / p).convert("RGB"))
    ax.set_title(["fresh", "rotten"][l], fontsize=10)
    ax.axis("off")
plt.suptitle("Sample training photos")
plt.tight_layout()
plt.savefig(OUT / "fig2_samples.png", dpi=150)
plt.close()

# Figure 3: photo sizes and brightness (random sample of 800 photos)
sample = random.sample(items, min(800, len(items)))
widths, heights, bright = [], [], {0: [], 1: []}
for p, l in sample:
    im = Image.open(DATA / p)
    widths.append(im.size[0])
    heights.append(im.size[1])
    bright[l].append(np.asarray(im.convert("L").resize((64, 64)), dtype=float).mean())
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(widths, heights, s=8, alpha=0.5)
ax[0].set_xlabel("width (pixels)")
ax[0].set_ylabel("height (pixels)")
ax[0].set_title("Original photo sizes (800 photos)")
ax[1].hist(bright[0], bins=25, alpha=0.6, label="fresh", color="#4C9F70")
ax[1].hist(bright[1], bins=25, alpha=0.6, label="rotten", color="#C8553D")
ax[1].set_xlabel("average brightness (0-255)")
ax[1].set_ylabel("number of photos")
ax[1].set_title("Brightness by class")
ax[1].legend()
plt.tight_layout()
plt.savefig(OUT / "fig3_size_brightness.png", dpi=150)
plt.close()
print("Saved 3 figures in evidence/")
