"""Data utilities shared by train / test / predict.

Labels come from folder names: folders starting with 'fresh' -> 0, 'rotten' -> 1.
The official train/test folders of the Kaggle dataset are IGNORED: all images are
merged, exact duplicates removed, then split ~70/15/15 with a fixed seed.
The split is GROUPED: all edited copies (rotated, flipped, shifted, noisy) of one
original photo stay in the same split, so no photo leaks from train into test.
"""
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_SIZE = 64
CLASSES = ["fresh", "rotten"]
EXTS = {".jpg", ".jpeg", ".png"}
SEED = 42
AUG_PREFIX = re.compile(
    r"^(rotated_by_\d+_|translation_|saltandpepper_|vertical_flip_|horizontal_flip_)+", re.I)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def group_key(rel_path):
    """All edited copies of one original photo share the same key."""
    p = Path(rel_path)
    return p.parent.name.lower() + "/" + AUG_PREFIX.sub("", p.name).lower()


def scan_images(data_dir, fruits=None):
    """Return (list of (relative_path, label), number_of_duplicates_removed)."""
    data_dir = Path(data_dir)
    seen, items, dups = set(), [], 0
    for p in sorted(data_dir.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        folder = p.parent.name.lower()
        if folder.startswith("fresh"):
            label = 0
        elif folder.startswith("rotten"):
            label = 1
        else:
            continue
        if fruits and not any(f in folder for f in fruits):
            continue
        digest = hashlib.md5(p.read_bytes()).hexdigest()
        if digest in seen:
            dups += 1
            continue
        seen.add(digest)
        items.append((p.relative_to(data_dir).as_posix(), label))
    return items, dups


def make_split(data_dir, split_path, fruits=None):
    """Create (or load) a fixed ~70/15/15 split, grouped by original photo."""
    split_path = Path(split_path)
    if split_path.exists():
        return json.loads(split_path.read_text())
    from sklearn.model_selection import GroupShuffleSplit  # training env only

    items, dups = scan_images(data_dir, fruits)
    if len(items) < 100:
        raise SystemExit(f"Only {len(items)} images found in {data_dir}. Check the dataset folder.")
    paths = [p for p, _ in items]
    labels = [l for _, l in items]
    groups = [group_key(p) for p in paths]

    tr_idx, tmp_idx = next(GroupShuffleSplit(1, test_size=0.30, random_state=SEED).split(paths, labels, groups))
    tmp_idx = list(tmp_idx)
    va_rel, te_rel = next(GroupShuffleSplit(1, test_size=0.50, random_state=SEED).split(
        tmp_idx, [labels[i] for i in tmp_idx], [groups[i] for i in tmp_idx]))
    va_idx = [tmp_idx[i] for i in va_rel]
    te_idx = [tmp_idx[i] for i in te_rel]

    g_tr = {groups[i] for i in tr_idx}
    g_va = {groups[i] for i in va_idx}
    g_te = {groups[i] for i in te_idx}
    assert not (g_tr & g_va) and not (g_tr & g_te) and not (g_va & g_te), "leak: a photo is in two splits"

    def pack(idx):
        return [(paths[i], labels[i]) for i in idx]

    split = {"fruits": fruits, "classes": CLASSES, "duplicates_removed": dups,
             "n_groups": len(set(groups)),
             "train": pack(tr_idx), "val": pack(va_idx), "test": pack(te_idx)}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split))
    return split


def load_image_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> float tensor (3, 64, 64) scaled to 0-1."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class FruitDataset(Dataset):
    def __init__(self, data_dir, items, augment=None):
        self.data_dir = Path(data_dir)
        self.items = items
        self.augment = augment  # optional torchvision transform on PIL image
        key = hashlib.md5("|".join(p for p, _ in items).encode()).hexdigest()[:10]
        cache_file = self.data_dir / f"cache_{IMG_SIZE}_{key}.npy"
        self.images = None
        if cache_file.exists():
            try:
                self.images = np.load(cache_file)
            except (OSError, ValueError, EOFError):
                self.images = None  # damaged cache: rebuild it
        if self.images is None:
            print(f"Building image cache for {len(items)} images (one time only)...", flush=True)
            self.images = np.stack([
                np.asarray(Image.open(self.data_dir / p).convert("RGB")
                           .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR), dtype=np.uint8)
                for p, _ in items])
            try:
                tmp = cache_file.with_name(cache_file.stem + ".tmp.npy")
                np.save(tmp, self.images)
                tmp.replace(cache_file)
            except OSError:
                pass  # read-only folder: just skip saving

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.fromarray(self.images[i])
        if self.augment is not None:
            img = self.augment(img)
        return load_image_tensor(img), torch.tensor(float(self.items[i][1]))