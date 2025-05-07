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

            # Load allowed labels from config.yaml
            with open("configs/config.yaml", "r") as f:
                config = yaml.safe_load(f)

            self.config = config
            self.condition_type = config["condition_type"]
            self.image_root = image_root

            # Standardize to lowercase
            self.condition_type = config["condition_type"]
            self.df[config["mood_column"]] = self.df[config["mood_column"]].str.lower()
            self.df[config["color_column"]] = self.df[config["color_column"]].str.lower()


            #self.all_conditions = sorted(config[f"{condition_type}_labels"])
            if config["condition_type"] == "mood":
                self.all_conditions = sorted(config["mood_labels"])
            elif config["condition_type"] == "dual":
                self.all_conditions = sorted(config["mood_labels"] + config["color_labels"])



        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = row["filepath"]
            image = load_image(img_path)

            if self.condition_type == "mood":
                moood = row[self.config["mood_column"]]
                condition = one_hot_encode(mood, self.config["mood_labels"])

            elif self.condition_type == "dual":
                mood = row[self.config["mood_column"]]
                color = row[self.config["color_column"]]
                condition = get_condition_vector_dual(mood, color, self.config)

            else:
                raise ValueError(f"Unknown condition_type: {self.condition_type}")

            return image, condition
            """image_id = str(row["image_id"])
            condition_value = row[self.condition_type]
            img_path = os.path.join(self.image_root, f"{image_id}.jpg")
            img_tensor = load_image(img_path)
            #print(f"[DEBUG] Image shape: {img_tensor.shape}")  # Should be [3, 64, 64]
            #print(f"[DEBUG] Flattened shape: {img_tensor.view(-1).shape[0]}")  # Should be 12288
            condition_tensor = one_hot_encode(condition_value, self.all_conditions)
            return img_tensor, condition_tensor"""

    return MoodboardFunctionDataset()



def get_condition_vector_dual(mood, color, config):
    from utils.dataset import one_hot_encode
    mood_vector = one_hot_encode(mood, config["mood_labels"])
    color_vector = one_hot_encode(color, config["color_labels"])
    return torch.cat((mood_vector, color_vector), dim=0)  # size: [mood_dim + color_dim]