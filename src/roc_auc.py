import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def normalize_label(x):
    return x.strip().lower() if isinstance(x, str) else x


def main():
    gt_path = "data/test_images/test_metadata.csv"
    pred_path = "data/test_images/test_predictions2.csv"

    # Load CSVs
    df_gt = pd.read_csv(gt_path)
    df_pred = pd.read_csv(pred_path)

    # Clean column names
    df_gt.columns = df_gt.columns.str.strip().str.lower()
    df_pred.columns = df_pred.columns.str.strip().str.lower()

    # Normalize filenames (CRITICAL)
    df_gt["image_filename"] = df_gt["image_filename"].str.strip()
    df_pred["image_filename"] = df_pred["image_filename"].str.strip()

    # Normalize labels
    df_gt["classification"] = df_gt["classification"].apply(normalize_label)

    # Build lookup dict: filename → class
    gt_map = dict(
        zip(df_gt["image_filename"], df_gt["classification"])
    )

    # Collect aligned labels & probabilities
    y_true = []
    y_prob = []

    label_map = {"normal": 0, "benign": 1, "malignant": 2}

    for _, row in df_pred.iterrows():
        fname = row["image_filename"]

        if fname not in gt_map:
            continue

        label = gt_map[fname]
        if label not in label_map:
            continue

        y_true.append(label_map[label])
        y_prob.append([
            row["prob_normal"],
            row["prob_benign"],
            row["prob_malignant"]
        ])

    print("Aligned samples:", len(y_true))

    # Convert to arrays
    y_true = pd.Series(y_true).values
    y_prob = pd.DataFrame(y_prob).values

    # One-hot encode
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    # ROC–AUC (OvR)
    auc = roc_auc_score(y_true_bin, y_prob, multi_class="ovr")

    print(f"ROC–AUC (OvR, multi-class): {auc:.4f}")


if __name__ == "__main__":
    main()
