import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def normalize_label(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x

def main():
    gt_path = "data/test_images/test_metadata.csv"
    # pred_path = "data/test_images/test_predictions1.csv"
    pred_path = "data/test_images/test_predictions2.csv"

    df_gt = pd.read_csv(gt_path)
    df_pred = pd.read_csv(pred_path)

    # --- FIX 1: Clean Column Headers ---
    # This removes hidden spaces (e.g., "classification " -> "classification")
    df_gt.columns = df_gt.columns.str.strip()
    df_pred.columns = df_pred.columns.str.strip()

    print("Ground Truth Columns found:", df_gt.columns.tolist())
    
    # Normalize labels
    # Ensure 'classification' exists before applying
    if "classification" not in df_gt.columns:
        print("ERROR: 'classification' column not found in Ground Truth CSV.")
        print("Available columns:", df_gt.columns.tolist())
        return

    df_gt["classification"] = df_gt["classification"].apply(normalize_label)
    df_pred["predicted_label"] = df_pred["predicted_label"].apply(normalize_label)

    # --- FIX 2: Merge with explicit suffixes ---
    # If 'classification' exists in both, this ensures we know which is which.
    # 'classification' from GT becomes 'classification_gt'
    # 'classification' from Pred becomes 'classification_pred'
    df = df_gt.merge(df_pred, on="Image_filename", how="inner", suffixes=('_gt', '_pred'))

    print(f"Merged {len(df)} rows. Columns in merged df: {df.columns.tolist()}")

    # --- FIX 3: Select the correct column ---
    # Check if the simple name exists, otherwise look for the suffixed version
    if "classification" in df.columns:
        y_true = df["classification"]
    elif "classification_gt" in df.columns:
        y_true = df["classification_gt"]
    else:
        raise KeyError("Could not find 'classification' or 'classification_gt' in merged dataframe.")

    y_pred = df["predicted_label"]

    # Accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    main()