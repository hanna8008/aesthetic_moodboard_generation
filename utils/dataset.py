# -----------------------------------------------------------
# Custom Dataset Loader for CVAE Training
# -----------------------------------------------------------
# This script loads image data and their corresponding condition labels (e.g. mood or color)
# from a structured folder and CSV file. It handles:
#
# 1. Reading the meteadata from a CSV file
# 2. Matching image IDs to their file paths based on mood and color folder structure
# 3. Preprocessing images (resizing, normalization)
# 4. Converting condition labels to one-hot encoded tensors
#
# The final output is a list of (image_tensor, condition_tensor) pairs, which can be
# used for training a conditional variational autoencoder (CVAE). This setup allows
# the model to learn how conditions influence generated images.

# --- Imports ---
import os
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# --- Define how to process each image ---
# 1. resize all images to the same size (64x64)
# 2. convert images to tensors (so PyTorch can use them)
# 3. normalize pixel values to be between -1 and 1 for better training
transform = transforms.Compose([
    #resize each image to 64x64 pixels, to ensure they're all the same
    transforms.Resize((64, 64)),
    #convert image to PyTorch format (C, H, W)
    transforms.ToTensor(),
    #scale pixel values from [0, 1] to [-1, 1]
    #transforms.Normalize([0.5], [0.5])
])


def load_image(img_path):
    """
    Lods an image from a file and paplies the preprocessing transform
    """
    #make sure the image has 3 color channels (Red, Green, Blue)
    image = Image.open(img_path).convert("RGB")
    #apply image processing steps
    return transform(image)


def one_hot_encode(label, all_labels):
    """
    Turns a label (like 'happy') into a one-hot tensor
    Example: If there are 3 moods and label is 'happy', it might become [0, 1, 0]
    """
    #find the position of this label in the list of all labels
    idx = all_labels.index(label)
    #create a vector with a 1 at the label's index and 0s elsewhere
    return F.one_hot(torch.tensor(idx), num_classes=16).float()


def load_dataset(csv_path, image_root, condition_type):
    """ 
    Loads all image files and their lables for training the model.

    Arguments:
        - csv_path: CSV file with image metadata and labels
        - image_root: folder where the images are stored
        - condition_type: label to use for conditioning (e.g., 'mood', or 'color_theme')

    Returns:
        - a list of (image_tensor, condition_tensor) pairs
    """
    #read the CSV file into the DataFrame
    df = pd.read_csv(csv_path)

    #filter out rows with missing images
    df["filepath"] = df["image_id"].apply(lambda x: os.path.join(image_root, f"{x}.jpg"))
    #keep only rows with actual images
    df = df[df["filepath"].apply(os.path.exists)]

    #get a list of all unique mood or color labels (whichever is being conditionined on)
    all_conditions = sorted(df[condition_type].unique())

    #list to be filled with image + label pairs
    dataset = []

    #keep track of how many images are missing
    missing_count = 0

    #go through every row in the CSV
    for _,row in df.iterrows():
        #get the image file name
        image_id = str(row['image_id'])
        #read the mood column
        mood = row['mood']
        #read the color column
        color = row['color_theme']
        #pick the column to be used as the label
        condition_value = row[condition_type]

        #build the full file path to the image (folder structure is: mood/color/image.jpg)
        img_path = os.path.join(image_root, f"{image_id}.jpg")
        #skip it if the file does not exist
        if not os.path.exists(img_path):
            missing_count += 1
            continue

        #load and transform the image into a tensor
        img_tensor = load_image(img_path)
        #turn the label (like 'happy') into a one-hot tensor
        condition_tensor = one_hot_encode(condition_value, all_conditions)

        #add the image and its label to the dataset
        dataset.append((img_tensor, condition_tensor))

    print(f"Missing images: {missing_count}")

    #return the complete dataset
    return dataset