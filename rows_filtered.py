import pandas as pd
import os

df = pd.read_csv("data/filtered_mood_color_dataset.csv")
image_root = "data/filtered_images"

# Make sure images exist
df["filepath"] = df["image_id"].apply(lambda x: os.path.join(image_root, f"{x}.jpg"))
df = df[df["filepath"].apply(os.path.exists)]

# Check valid mood
valid_moods = ['cozy', 'dreamy', 'romantic', 'minimalist', 'vibrant', 'vintage', 'natural', 'adventurous', 'playful']
df = df[df["mood"].isin(valid_moods)]

print("Final usable dataset size:", len(df))  # Should match 3883

