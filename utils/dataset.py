import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import yaml

# Image transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def load_image(img_path):
    image = Image.open(img_path).convert("RGB")
    return transform(image)

def one_hot_encode(label, all_labels):
    idx = all_labels.index(label)
    return F.one_hot(torch.tensor(idx), num_classes=len(all_labels)).float()

def load_dataset(csv_path, image_root, condition_type):
    class MoodboardFunctionDataset(Dataset):
        def __init__(self):
            self.df = pd.read_csv(csv_path)
            self.df["filepath"] = self.df["image_id"].apply(lambda x: os.path.join(image_root, f"{x}.jpg"))
            self.df = self.df[self.df["filepath"].apply(os.path.exists)]

            self.condition_type = condition_type
            self.image_root = image_root

            # Standardize to lowercase
            self.df[condition_type] = self.df[condition_type].str.lower()

            # Load allowed labels from config.yaml
            with open("configs/config.yaml", "r") as f:
                config = yaml.safe_load(f)

            self.all_conditions = sorted(config[f"{condition_type}_labels"])


        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_id = str(row["image_id"])
            condition_value = row[self.condition_type]
            img_path = os.path.join(self.image_root, f"{image_id}.jpg")
            img_tensor = load_image(img_path)
            condition_tensor = one_hot_encode(condition_value, self.all_conditions)
            return img_tensor, condition_tensor

    return MoodboardFunctionDataset()