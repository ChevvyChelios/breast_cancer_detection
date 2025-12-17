
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

from src.dataset import get_mask_bbox
from src.model import create_model


# ---------------- DATASET ----------------
class TestDataset(Dataset):
    def __init__(self, csv_path, root_main, root_mask):
        self.df = pd.read_csv(csv_path)
        self.root_main = root_main
        self.root_mask = root_mask

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- MAIN IMAGE ----
        main_path = os.path.join(self.root_main, row["Image_filename"])
        main_img = Image.open(main_path).convert("RGB")

        # ---- MASK (HANDLE NaN) ----
        mask_name = row["Mask_tumor_filename"]
        mask = None
        if not pd.isna(mask_name):
            mask_path = os.path.join(self.root_mask, str(mask_name))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")

        # ---- ROI CROP ----
        roi_img = main_img
        if mask is not None:
            bbox = get_mask_bbox(mask)
            if bbox is not None:
                roi_img = main_img.crop(bbox)

        roi_img = self.transform(roi_img)
        return roi_img, idx


# ---------------- MAIN ----------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_csv = os.path.join("data", "test_images", "test_metadata.csv")
    main_root = os.path.join("data", "test_images", "main")
    mask_root = os.path.join("data", "test_images", "masked_tumor")

    df = pd.read_csv(test_csv)

    # ---- LABEL MAP ----
    with open(os.path.join("metadata", "label_mapping.json")) as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    dataset = TestDataset(test_csv, main_root, mask_root)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    # ---- LOAD MODEL ----
    model = create_model(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load("models/model2.pt", map_location=device))
    model.to(device)
    model.eval()

    # ---- STORAGE ----
    all_preds = [None] * len(df)
    all_conf = [None] * len(df)

    prob_normal = [None] * len(df)
    prob_benign = [None] * len(df)
    prob_malignant = [None] * len(df)

    # ---- INFERENCE ----
    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)

            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(probs, dim=1)

            probs = probs.cpu().numpy()
            preds = preds.cpu().numpy()

            for i, row_idx in enumerate(idxs.numpy()):
                all_preds[row_idx] = preds[i]
                all_conf[row_idx] = probs[i, preds[i]]

                prob_normal[row_idx] = probs[i, 0]
                prob_benign[row_idx] = probs[i, 1]
                prob_malignant[row_idx] = probs[i, 2]

    # ---- SAVE RESULTS ----
    df["predicted_label_idx"] = all_preds
    df["predicted_label"] = df["predicted_label_idx"].map(idx_to_label)
    df["predicted_confidence"] = all_conf

    df["prob_normal"] = prob_normal
    df["prob_benign"] = prob_benign
    df["prob_malignant"] = prob_malignant

    out_path = os.path.join("data", "test_images", "test_predictions2.csv")
    df.to_csv(out_path, index=False)

    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()
