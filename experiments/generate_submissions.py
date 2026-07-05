"""
Generate submission CSVs for all trained embedders.

Usage:
  On AMD box: run this after test_all_embedders.ipynb has trained all models
  Generates: submission_v1.csv, submission_v2.csv, submission_brain_id.csv
  Saves to: /shared-docker/mohamed/ and /workspace/out/
"""
import os
import sys
import csv
from pathlib import Path

import numpy as np

# Setup paths
DATA_ROOT = Path(os.environ.get("DATA_INPUT_ROOT", "/workspace/data/ehl"))
WORK_DIR = Path(os.environ.get("WORK_DIR", "/workspace/out"))
SHARED = Path("/shared-docker")
AMINE = SHARED / "amine"
MOHAMED = SHARED / "mohamed"

sys.path.insert(0, str(AMINE))
sys.path.insert(0, str(MOHAMED))

from eval_harness import build_image_index, cached_volume, read_csv, GRID


def generate_submission(embedder_name, embed_fn, output_path):
    """
    Generate submission CSV for a single embedder.

    Format: query_id,target_id_ranking
    where target_id_ranking is space-separated IDs ranked by similarity (highest first)
    """
    print(f"\n{'='*70}")
    print(f"Generating submission for {embedder_name.upper()}")
    print(f"{'='*70}")

    # Build index
    index = build_image_index(DATA_ROOT)
    print(f"Indexed {len(index)} images")

    # Load all pairs (for training reference)
    all_pairs = read_csv(DATA_ROOT / "dataset1" / "train_pairs.csv")

    # Load test queries and galleries for each dataset
    submission_rows = []

    for dataset_num in [1, 2, 3]:
        dataset_dir = DATA_ROOT / f"dataset{dataset_num}"

        # Load val pairs
        val_pairs = read_csv(dataset_dir / "val_pairs.csv")
        print(f"\nDataset {dataset_num}: {len(val_pairs)} val queries")

        # For each val query: rank all val galleries
        for query_pair in val_pairs:
            q_id = query_pair["query_id"]
            q_vol = cached_volume(q_id, index[q_id], GRID)
            q_emb = embed_fn(q_vol)

            # Rank all galleries
            g_ids = [p["target_id"] for p in val_pairs]
            g_embeds = np.array([embed_fn(cached_volume(g_id, index[g_id], GRID)) for g_id in g_ids])

            # Cosine similarity (already L2-normalized)
            sims = q_emb @ g_embeds.T
            ranked_indices = np.argsort(-sims)  # descending
            ranked_ids = [g_ids[i] for i in ranked_indices]

            submission_rows.append((q_id, " ".join(ranked_ids)))

        # Load test pairs
        test_pairs = read_csv(dataset_dir / "test_pairs.csv")
        print(f"Dataset {dataset_num}: {len(test_pairs)} test queries")

        for query_pair in test_pairs:
            q_id = query_pair["query_id"]
            q_vol = cached_volume(q_id, index[q_id], GRID)
            q_emb = embed_fn(q_vol)

            # Rank all galleries
            g_ids = [p["target_id"] for p in test_pairs]
            g_embeds = np.array([embed_fn(cached_volume(g_id, index[g_id], GRID)) for g_id in g_ids])

            sims = q_emb @ g_embeds.T
            ranked_indices = np.argsort(-sims)
            ranked_ids = [g_ids[i] for i in ranked_indices]

            submission_rows.append((q_id, " ".join(ranked_ids)))

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "target_id_ranking"])
        writer.writerows(submission_rows)

    print(f"\n✓ Wrote {len(submission_rows)} rows to {output_path}")
    return len(submission_rows)


if __name__ == "__main__":
    # Import embedders
    print("Loading embedders...")
    from learned_embedder import build as build_v1
    from learned_embedder_v2 import build as build_v2

    try:
        from brain_id_embedder import build as build_brain_id

        has_brain_id = True
    except Exception as e:
        has_brain_id = False
        print(f"Brain-ID not available: {e}")

    # Load training data (once)
    print("Loading training data...")
    train_pairs = read_csv(DATA_ROOT / "dataset1" / "train_pairs.csv")
    index = build_image_index(DATA_ROOT)

    np.random.seed(20260627)
    perm = np.arange(len(train_pairs))
    np.random.shuffle(perm)
    val_idx = set(perm[:60])
    train_pairs_split = [p for i, p in enumerate(train_pairs) if i not in val_idx]

    # Train embedders
    print("\nTraining embedders...")
    embedders = {}

    print("\n1. Training v1...")
    embedders["v1"] = build_v1(train_pairs_split, index, GRID, cached_volume)

    print("\n2. Training v2...")
    embedders["v2"] = build_v2(train_pairs_split, index, GRID, cached_volume)

    if has_brain_id:
        print("\n3. Training Brain-ID...")
        try:
            embedders["brain_id"] = build_brain_id(train_pairs_split, index, GRID, cached_volume)
        except Exception as e:
            print(f"Brain-ID training failed: {e}")

    # Generate submissions
    print("\n\nGenerating submissions...")
    output_dir = WORK_DIR / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, embed_fn in embedders.items():
        output_path = output_dir / f"submission_{name}.csv"
        try:
            generate_submission(name, embed_fn, output_path)
        except Exception as e:
            print(f"❌ Failed to generate {name}: {e}")

    print(f"\n{'='*70}")
    print(f"All submissions saved to {output_dir}")
    print(f"Ready to download and submit to Kaggle!")
    print(f"{'='*70}")
