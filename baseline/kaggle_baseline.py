"""
Kaggle-notebook version of the slice-CLIP baseline for the
Brain MRI Cross-Modal Retrieval Challenge.

HOW TO USE ON KAGGLE
--------------------
1. Open the competition page -> "Code" -> "New Notebook".
   (This auto-attaches the competition data at
    /kaggle/input/ehl-paris-medical-image-retrieval/)
2. Settings -> Accelerator -> GPU (P100 or T4).
3. Settings -> Internet -> ON  (needed for the pip install below).
4. Paste this whole file into one cell and Run All.
5. It writes /kaggle/working/submission.csv.
6. Submit: either "Submit to Competition" (top-right of the notebook),
   or download submission.csv and upload it on the competition page.

This is the unmodified baseline logic (tiny 2D-slice CLIP), only the
config/paths/device are adapted for Kaggle. It is our REFERENCE score,
not a strong solution.
"""

# --- Setup: install MONAI (torch is already present) ------------------------
# On Kaggle we pip-install MONAI/nibabel at runtime. On other machines (e.g. the
# AMD MI300X box) install deps yourself first so pip can't clobber the ROCm torch
# build -- see the run instructions. INSTALL_DEPS defaults to on only on Kaggle;
# force it anywhere with INSTALL_DEPS=1.
import os
import subprocess, sys

if os.environ.get("INSTALL_DEPS", "1" if os.path.isdir("/kaggle/input") else "0") == "1":
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "monai>=1.5.0", "nibabel>=5.3"], check=True)

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import PersistentDataset
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    Orientationd,
    RandGaussianNoised,
    ScaleIntensityd,
    Spacingd,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --- Paths / device adapted for Kaggle --------------------------------------
# The competition CSVs store ABSOLUTE image paths baked in from the organizers'
# environment (e.g. /kaggle/input/competitions/.../foo.nii.gz) and the listed
# extension (.nii.gz) does not match the actual files (.nii). So we ignore the
# stored paths entirely and resolve every image by its ID via a glob index.
# Override these on non-Kaggle machines (e.g. the AMD box) via env vars:
#   DATA_INPUT_ROOT  -> folder to glob for dataset1/2/3 (default /kaggle/input)
#   WORK_DIR         -> output + MONAI cache dir   (default /kaggle/working)
INPUT_ROOT = Path(os.environ.get("DATA_INPUT_ROOT", "/kaggle/input"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/kaggle/working"))
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = WORK_DIR / "submission.csv"


def _find_data_root() -> Path:
    """Locate the folder that contains dataset1/2/3 under /kaggle/input."""
    hits = list(INPUT_ROOT.glob("**/dataset1/train_pairs.csv"))
    if not hits:
        raise FileNotFoundError(
            "Could not find dataset1/train_pairs.csv under /kaggle/input. "
            "Attach the competition data via '+ Add Input'."
        )
    return hits[0].parent.parent


DATA_ROOT = _find_data_root()
print("Detected DATA_ROOT:", DATA_ROOT)


def _build_image_index(root: Path) -> dict[str, str]:
    """Map every image ID (filename without .nii/.nii.gz) to its real path."""
    index: dict[str, str] = {}
    for p in root.glob("**/*.nii*"):
        name = p.name
        if name.endswith(".nii.gz"):
            stem = name[:-7]
        elif name.endswith(".nii"):
            stem = name[:-4]
        else:
            continue
        index[stem] = str(p)
    return index


IMAGE_INDEX = _build_image_index(DATA_ROOT)
print(f"Indexed {len(IMAGE_INDEX)} image files.")

# Every (query_csv, gallery_csv) pair to rank. Same-dataset, same-split only.
PREDICTION_SETS = [
    ("dataset1/val_queries.csv",  "dataset1/val_gallery.csv"),
    ("dataset1/test_queries.csv", "dataset1/test_gallery.csv"),
    ("dataset2/val_queries.csv",  "dataset2/val_gallery.csv"),
    ("dataset2/test_queries.csv", "dataset2/test_gallery.csv"),
    ("dataset3/val_queries.csv",  "dataset3/val_gallery.csv"),
    ("dataset3/test_queries.csv", "dataset3/test_gallery.csv"),
]
TRAIN_PAIR_CSV = "dataset1/train_pairs.csv"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


CONFIG = {
    "seed": 20260626,
    "cache_dir": str(WORK_DIR / ".monai_persistent"),
    "spacing_mm": (1.0, 1.0, 1.0),
    "slice_positions": (0.35, 0.50, 0.65),
    "image_size": 96,
    "noise_probability": 0.3,
    "noise_std": 0.05,
    "epochs": 500,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "embedding_dim": 128,
    "encoder_hidden_dim": 512,
    "similarity_scale": 5.0,
    "max_grad_norm": 1.0,
    "num_workers": 2,
    "device": pick_device(),
}


class SliceStackd(MapTransform):
    """MONAI map transform for three representative 2D slices."""

    def __init__(self, keys, positions, image_size):
        super().__init__(keys)
        self.positions = positions
        self.image_size = image_size

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            volume = torch.as_tensor(d[key]).float()
            if volume.ndim != 4:
                raise ValueError(f"Expected channel-first 3D volume, got {tuple(volume.shape)}")
            volume = volume[0]
            finite = torch.where(torch.isfinite(volume), volume, torch.zeros_like(volume))
            nonzero_counts = torch.count_nonzero(finite, dim=(0, 1))
            occupied = torch.nonzero(nonzero_counts, as_tuple=False).flatten()
            if len(occupied) == 0:
                z_values = [volume.shape[-1] // 2] * len(self.positions)
            else:
                z_min = int(occupied[0])
                z_max = int(occupied[-1])
                z_values = [round(z_min + p * (z_max - z_min)) for p in self.positions]
            slices = torch.stack([volume[:, :, int(np.clip(z, 0, volume.shape[-1] - 1))] for z in z_values])
            slices = torch.nan_to_num(slices, nan=0.0, posinf=0.0, neginf=0.0)
            slices = F.interpolate(
                slices.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            d[key] = slices
        return d


class TinySliceEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        pooled = int(CONFIG["image_size"]) // 8
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled * pooled, int(CONFIG["encoder_hidden_dim"])),
            nn.ReLU(),
            nn.Linear(int(CONFIG["encoder_hidden_dim"]), embedding_dim),
        )

    def forward(self, x):
        return self.projection(self.features(x))


class SliceCLIP(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.query_encoder = TinySliceEncoder(embedding_dim)
        self.target_encoder = TinySliceEncoder(embedding_dim)
        self.similarity_scale = float(CONFIG["similarity_scale"])

    def forward(self, query, target):
        q = self.encode_query(query)
        t = self.encode_target(target)
        return self.similarity_scale * q @ t.T

    def encode_query(self, query):
        return F.normalize(self.query_encoder(query), dim=1)

    def encode_target(self, target):
        return F.normalize(self.target_encoder(target), dim=1)


class PairImageDataset(Dataset):
    def __init__(self, pairs, image_dataset):
        self.image_dataset = image_dataset
        self.id_to_index = {row["id"]: i for i, row in enumerate(image_dataset.data)}
        self.examples = [(str(p["query_id"]), str(p["target_id"])) for p in pairs]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        q, t = self.examples[index]
        return self._image(q), self._image(t)

    def _image(self, image_id):
        item = self.image_dataset[self.id_to_index[image_id]]
        return torch.as_tensor(item["image"]).float()


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolve_id(image_id):
    """Resolve an image ID to its real file path via the glob index."""
    path = IMAGE_INDEX.get(image_id)
    if path is None:
        raise KeyError(f"Image ID {image_id!r} not found among {len(IMAGE_INDEX)} indexed files.")
    return Path(path)


def load_manifest_images(csv_path, id_column, image_column):
    # image_column is ignored on purpose; we resolve by ID through IMAGE_INDEX.
    images = {}
    for row in read_csv(csv_path):
        images[row[id_column]] = resolve_id(row[id_column])
    return images


def load_training_pairs():
    pairs = []
    for row in read_csv(DATA_ROOT / TRAIN_PAIR_CSV):
        pairs.append({
            "query_id": row["query_id"],
            "target_id": row["target_id"],
            "query_path": resolve_id(row["query_id"]),
            "target_path": resolve_id(row["target_id"]),
        })
    if not pairs:
        raise ValueError("No training pairs found.")
    return pairs


def collect_prediction_sets():
    sets = []
    for q_csv, g_csv in PREDICTION_SETS:
        queries = load_manifest_images(DATA_ROOT / q_csv, "query_id", "query_image")
        targets = load_manifest_images(DATA_ROOT / g_csv, "target_id", "target_image")
        sets.append({"queries": queries, "targets": targets})
    return sets


def monai_transform(augment):
    deterministic = [
        LoadImaged(keys="image", image_only=True),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS", labels=None),
        Spacingd(keys="image", pixdim=CONFIG["spacing_mm"], mode="bilinear"),
        ScaleIntensityd(keys="image", minv=0.0, maxv=1.0),
        SliceStackd(keys="image", positions=tuple(CONFIG["slice_positions"]),
                    image_size=int(CONFIG["image_size"])),
        EnsureTyped(keys="image"),
    ]
    aug = []
    if augment:
        aug.append(RandGaussianNoised(keys="image", prob=float(CONFIG["noise_probability"]),
                                      mean=0.0, std=float(CONFIG["noise_std"])))
    return Compose(deterministic + aug)


def make_image_dataset(images, augment):
    cache_dir = Path(CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"image": str(path), "id": iid} for iid, path in sorted(images.items())]
    return PersistentDataset(data=rows, transform=monai_transform(augment), cache_dir=cache_dir)


def train_model(train_dataset):
    torch.manual_seed(int(CONFIG["seed"]))
    device = torch.device(str(CONFIG["device"]))
    model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(CONFIG["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(train_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=True,
                        num_workers=int(CONFIG["num_workers"]))
    model.train()
    for epoch in range(1, int(CONFIG["epochs"]) + 1):
        total_loss, total_seen = 0.0, 0
        for q, t in loader:
            q, t = q.to(device), t.to(device)
            labels = torch.arange(len(q), device=device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(q, t)
            loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) / 2
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(CONFIG["max_grad_norm"]))
            optimizer.step()
            total_loss += float(loss.item()) * len(q)
            total_seen += len(q)
        if epoch % 25 == 0 or epoch == 1:
            print(f"epoch {epoch:03d} loss={total_loss / max(total_seen, 1):.5f}")
    return model


@torch.no_grad()
def embed_images(model, image_dataset, encoder):
    device = next(model.parameters()).device
    loader = DataLoader(image_dataset, batch_size=int(CONFIG["batch_size"]), shuffle=False)
    embeddings = {}
    model.eval()
    for batch in tqdm(loader, desc=f"Embedding ({encoder})"):
        images = torch.as_tensor(batch["image"]).float().to(device)
        if encoder == "query":
            emb = model.encode_query(images).cpu().numpy().astype(np.float32)
        else:
            emb = model.encode_target(images).cpu().numpy().astype(np.float32)
        for iid, e in zip(batch["id"], emb):
            embeddings[str(iid)] = e
    return embeddings


@torch.no_grad()
def rank_targets(model, query_embeddings, target_embeddings):
    device = next(model.parameters()).device
    rows = []
    target_ids = sorted(target_embeddings)
    target_matrix = torch.from_numpy(np.stack([target_embeddings[t] for t in target_ids])).to(device)
    for qid in sorted(query_embeddings):
        q = torch.from_numpy(query_embeddings[qid]).to(device)
        scores = (model.similarity_scale * q.unsqueeze(0) @ target_matrix.T).squeeze(0).cpu().numpy()
        ranking = [target_ids[i] for i in np.argsort(-scores)]
        rows.append({"query_id": qid, "target_id_ranking": " ".join(ranking)})
    return rows


def write_submission(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    random.seed(int(CONFIG["seed"]))
    np.random.seed(int(CONFIG["seed"]))

    train_pairs = load_training_pairs()
    prediction_sets = collect_prediction_sets()

    train_images = {}
    for p in train_pairs:
        train_images[str(p["query_id"])] = Path(p["query_path"])
        train_images[str(p["target_id"])] = Path(p["target_path"])

    inference_images = {}
    for ps in prediction_sets:
        inference_images.update(ps["queries"])
        inference_images.update(ps["targets"])

    print(json.dumps({
        "device": CONFIG["device"],
        "num_train_images": len(train_images),
        "num_inference_images": len(inference_images),
        "num_train_pairs": len(train_pairs),
    }, indent=2))

    train_image_dataset = make_image_dataset(train_images, augment=True)
    inference_image_dataset = make_image_dataset(inference_images, augment=False)
    train_dataset = PairImageDataset(train_pairs, train_image_dataset)

    model = train_model(train_dataset)
    query_emb = embed_images(model, inference_image_dataset, "query")
    target_emb = embed_images(model, inference_image_dataset, "target")

    submission_rows = []
    for ps in prediction_sets:
        q = {i: query_emb[i] for i in ps["queries"]}
        t = {i: target_emb[i] for i in ps["targets"]}
        submission_rows.extend(rank_targets(model, q, t))

    write_submission(OUT_PATH, submission_rows)
    print(f"Wrote {len(submission_rows)} rows to {OUT_PATH}")


main()
