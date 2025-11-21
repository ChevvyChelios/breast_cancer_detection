# src/create_splits.py
import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATA_ROOT = os.path.join("data", "train_images")
META_DIR = "metadata"
os.makedirs(META_DIR, exist_ok=True)

CLASS_NAMES = ["normal", "benign", "malignant"]  # define order explicitly


def collect_samples():
    rows = []
    for label_name in CLASS_NAMES:
        main_dir = os.path.join(DATA_ROOT, label_name, "main")
        for fname in os.listdir(main_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            rel_image_path = os.path.join("data", "train_images", label_name, "main", fname)
            rows.append({"image_path": rel_image_path, "label_name": label_name})
    return pd.DataFrame(rows)


def main():
    df = collect_samples()
    print("Total samples:", len(df))
    print(df["label_name"].value_counts())

    # Map label_name -> int
    label_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    df["label"] = df["label_name"].map(label_to_idx)

    # Stratified train/val split (80/20)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_path = os.path.join(META_DIR, "train.csv")
    val_path = os.path.join(META_DIR, "val.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    print("Saved:", train_path, "and", val_path)

    # Save label mapping
    mapping_path = os.path.join(META_DIR, "label_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(label_to_idx, f, indent=2)
    print("Saved label mapping to", mapping_path)


if __name__ == "__main__":
    main()
