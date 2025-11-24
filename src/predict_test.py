# src/predict_test.py
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

class TestDataset(Dataset):
    def __init__(self, csv_path, root_main, root_mask):
        self.df = pd.read_csv(csv_path)
        self.root_main = root_main
        self.root_mask = root_mask

        self.tform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    def __len__(self):
        return len(self.df)    

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        main_path = os.path.join(self.root_main, row["Image_filename"])
        main_img = Image.open(main_path).convert("RGB")

        # mask_path = os.path.join(self.root_mask, row["Mask_tumor_filename"])
        # ---- MASK IMAGE (Handle NaN) ----
        mask_name = row["Mask_tumor_filename"]
        # If mask is NaN → no mask available
        if pd.isna(mask_name):
            mask = None
        else:
            mask_path = os.path.join(self.root_mask, str(mask_name))
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = None

        roi_img = main_img
        if mask is not None:
            bbox = get_mask_bbox(mask)
            if bbox is not None:
                roi_img = main_img.crop(bbox)

        roi_img = self.tform(roi_img)
        return roi_img, idx



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    test_csv_path = os.path.join("data", "test_images", "test_metadata.csv")  # your actual name
    images_root = os.path.join("data", "test_images", "main")

    df = pd.read_csv(test_csv_path)

    # Load label mapping to restore names in correct order
    mapping_path = os.path.join("metadata", "label_mapping.json")
    with open(mapping_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    mask_root = os.path.join("data", "test_images", "masked_tumor")
    dataset = TestDataset(test_csv_path, images_root, mask_root)
    # dataset = TestUltrasoundDataset(test_csv_path, images_root)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    model = create_model(num_classes=3, pretrained=False)
    best_model_path = os.path.join("models", "model2.pt")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = [None] * len(df)
    all_probs = [None] * len(df)

    with torch.no_grad():
        for imgs, idxs in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)

            preds = preds.cpu().tolist()
            confs = confs.cpu().tolist()

            for i, row_idx in enumerate(idxs.tolist()):
                all_preds[row_idx] = preds[i]
                all_probs[row_idx] = confs[i]

    df["predicted_label_idx"] = all_preds
    df["predicted_label"] = df["predicted_label_idx"].map(idx_to_label)
    df["predicted_confidence"] = all_probs

    out_path = os.path.join("data", "test_images", "test_predictions2.csv")
    df.to_csv(out_path, index=False)
    print("Saved predictions to:", out_path)


if __name__ == "__main__":
    main()


# src/predict_test.py
# import os
# import json

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import pandas as pd

# from src.model import create_model


# class TestUltrasoundDataset(Dataset):
#     def __init__(self, csv_path, images_root):
#         self.df = pd.read_csv(csv_path)
#         self.images_root = images_root

#         self.transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             ),
#         ])

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_name = row["Image_filename"]   # adjust if column name differs
#         img_path = os.path.join(self.images_root, img_name)

#         img = Image.open(img_path).convert("RGB")
#         img = self.transform(img)
#         return img, idx  # return idx so we can map back


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     test_csv_path = os.path.join("data", "test_images", "test_metadata.csv")  # your actual name
#     images_root = os.path.join("data", "test_images", "main")

#     df = pd.read_csv(test_csv_path)

#     # Load label mapping to restore names in correct order
#     mapping_path = os.path.join("metadata", "label_mapping.json")
#     with open(mapping_path, "r") as f:
#         label_to_idx = json.load(f)
#     idx_to_label = {v: k for k, v in label_to_idx.items()}

#     dataset = TestUltrasoundDataset(test_csv_path, images_root)
#     loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

#     model = create_model(num_classes=3, pretrained=False)
#     best_model_path = os.path.join("models", "model1.pt")
#     model.load_state_dict(torch.load(best_model_path, map_location=device))
#     model = model.to(device)
#     model.eval()

#     all_preds = [None] * len(df)
#     all_probs = [None] * len(df)

#     with torch.no_grad():
#         for imgs, idxs in loader:
#             imgs = imgs.to(device)
#             outputs = model(imgs)
#             probs = F.softmax(outputs, dim=1)
#             confs, preds = torch.max(probs, 1)

#             preds = preds.cpu().tolist()
#             confs = confs.cpu().tolist()

#             for i, row_idx in enumerate(idxs.tolist()):
#                 all_preds[row_idx] = preds[i]
#                 all_probs[row_idx] = confs[i]

#     df["predicted_label_idx"] = all_preds
#     df["predicted_label"] = df["predicted_label_idx"].map(idx_to_label)
#     df["predicted_confidence"] = all_probs

#     out_path = os.path.join("data", "test_images", "test_predictions1.csv")
#     df.to_csv(out_path, index=False)
#     print("Saved predictions to:", out_path)


# if __name__ == "__main__":
#     main()
