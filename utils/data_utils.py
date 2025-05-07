# -------------------------------------------------------
# Custom PyTorch Dataset and Condition Vector Utilities
# -------------------------------------------------------
#
# Loads image data and labels (mood, color) from a CSV file, 
# applies transformations, and returns each sample as an image tensor
# with a one-hot endcoded condition vector
#
# Design to support flexible conditioning:
# - mood-only
# - dual conditioning (mood + color)
#
# Also includes helper functions for loading images and encoding labels,
# ensuring consistent preprocessing across training and generation scripts.



# --- Imports ---
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
#from utils.data_utils import one_hot_encode
from PIL import Image
import torch.nn.functional as F
import yaml



# --- Image Transformation ---
#define a transformation pipeline to preprocess images before feeding them to the model
transform = transforms.Compose([
    #resize all images to 64x64 pixels for model consistency
    transforms.Resize((64, 64)),
    #convert image to PyTorch tensor and scale pixel values to [0, 1]
    transforms.ToTensor(),
])



# -- Load Image ---
def load_image(img_path):
    #open image and ensure it's in RGB format
    image = Image.open(img_path).convert("RGB")
    #apply the predefined transform (resize + tensor)
    return transform(image)



# --- One Hot Encoding ---
def one_hot_encode(label, all_labels):
    #get the index of the label in the list of all labels
    idx = all_labels.index(label)
    #return one-hot encoded vector as float tensor
    return F.one_hot(torch.tensor(idx), num_classes=len(all_labels)).float()



# --- Load Dataset ---
#takes path to CSV, image folder, and condition type (mood or dual)
def load_dataset(csv_path, image_root, condition_type):
    #define a custom dataset class
    class MoodboardFunctionDataset(Dataset):

        def __init__(self):
            #load the CSV metadata into a DataFrame
            self.df = pd.read_csv(csv_path)

            #create full image file paths for each image_id
            self.df["filepath"] = self.df["image_id"].apply(lambda x: os.path.join(image_root, f"{x}.jpg"))
            
            #filter out any rows whwere the image file doesn't exist
            self.df = self.df[self.df["filepath"].apply(os.path.exists)]

            #load config.yaml to access labels, column names, and condition settings
            with open("configs/config.yaml", "r") as f:
                config = yaml.safe_load(f)

            #store the full config in the object
            self.config = config
            #e.g., "mood" or "dual"
            self.condition_type = config["condition_type"]
            #store image root path
            self.image_root = image_root

            #standardize condition column entries to lowercase for consistency
            self.df[config["mood_column"]] = self.df[config["mood_column"]].str.lower()
            self.df[config["color_column"]] = self.df[config["color_column"]].str.lower()

            #define valid condition values based on type (mood only or mood + color)
            if config["condition_type"] == "mood":
                #store only mood labels
                self.all_conditions = sorted(config["mood_labels"])
            elif config["condition_type"] == "dual":
                #concatenate and sort both mood and color labels
                self.all_conditions = sorted(config["mood_labels"] + config["color_labels"])



        # --- Dataset Length ---
        def __len__(self):
            #return total number of samples (images)
            return len(self.df)



        # --- What Happens on Each Data Access (via DataLoader) ---
        def __getitem__(self, idx):
            #get row corresponding to the sample index
            row = self.df.iloc[idx]
            #get the full image path
            img_path = row["filepath"]
            #load and transform the image
            image = load_image(img_path)

            #if using mood-only conditioning
            if self.condition_type == "mood":
                #extract mood value
                moood = row[self.config["mood_column"]]
                #convert mood to one-hot vector
                condition = one_hot_encode(mood, self.config["mood_labels"])

            #if using both mood + color
            elif self.condition_type == "dual":
                #extract mood value
                mood = row[self.config["mood_column"]]
                #extract color value
                color = row[self.config["color_column"]]
                #combine both into one condition vector
                condition = get_condition_vector_dual(mood, color, self.config)

            else:
                #handle unsupported condition types
                raise ValueError(f"Unknown condition_type: {self.condition_type}")

            #return image tensor and condition tensor to the model
            return image, condition

    #instantiate and return your custom dataset class
    return MoodboardFunctionDataset()



# --- Combine Mood and Color into One Vector ---
def get_condition_vector_dual(mood, color, config):
    #encode mood
    mood_vector = one_hot_encode(mood, config["mood_labels"])
    #encode color
    color_vector = one_hot_encode(color, config["color_labels"])
    #concatenate into a single condition vector
    return torch.cat((mood_vector, color_vector), dim=0)  # size: [mood_dim + color_dim]