import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def get_mask_bbox(mask_img):
    mask = np.array(mask_img)
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    return (x_min, y_min, x_max, y_max)


class BreastUltrasoundDataset(Dataset):
    def __init__(self, csv_path, augment=False):
        self.df = pd.read_csv(csv_path)

        self.transform_main = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        self.transform_roi = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        self.augment = augment
        if self.augment:
            self.aug = transforms.RandomHorizontalFlip()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        main_img = Image.open(row["main_image"]).convert("RGB")

        # Load mask
        mask_path = row["mask_image"]
        mask = None
        if isinstance(mask_path, str) and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert("L")

        # Compute ROI crop
        roi_img = main_img
        if mask is not None:
            bbox = get_mask_bbox(mask)
            if bbox is not None:
                roi_img = main_img.crop(bbox)

        # Apply augment
        if self.augment:
            main_img = self.aug(main_img)
            roi_img = self.aug(roi_img)

        # Transform
        main_img = self.transform_main(main_img)
        roi_img  = self.transform_roi(roi_img)

        # Option 1: Train only on ROI → best
        return roi_img, int(row["label"])

        # Option 2: Use both ROI + full image (two-stream)
        # return torch.cat([main_img, roi_img], dim=0), int(row["label"])

    def __len__(self):
        return len(self.df)


# src/dataset.py
# import os
# import pandas as pd
# from PIL import Image

# import torch
# from torch.utils.data import Dataset
# from torchvision import transforms


# class BreastUltrasoundDataset(Dataset):
#     def __init__(self, csv_path, augment=False):
#         self.df = pd.read_csv(csv_path)

#         self.label_col = "label"
#         self.image_col = "image_path"

#         if augment:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomRotation(10),
#                 transforms.RandomAffine(
#                     degrees=0,
#                     translate=(0.05, 0.05),
#                     scale=(0.95, 1.05)
#                 ),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 ),
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 ),
#             ])

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         img_path = row[self.image_col]
#         label = int(row[self.label_col])

#         # Make sure this is a correct path relative to project root
#         img = Image.open(img_path).convert("RGB")

#         img = self.transform(img)
#         return img, label
