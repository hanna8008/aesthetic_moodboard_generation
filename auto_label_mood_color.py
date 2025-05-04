import os
import json
import pandas as pd
import shutil
from collections import defaultdict

# --- File Paths ---
ATTRIBUTES_PATH = "data/pexels-110k-512p-min-jpg/attributes_df.json"
IMAGE_ROOT = "data/pexels-110k-512p-min-jpg/images"
OUTPUT_CSV = "data/filtered_mood_color_dataset.csv"
OUTPUT_IMAGE_DIR = "data/filtered_images"

# --- Load JSON ---
with open(ATTRIBUTES_PATH, "r") as f:
    attributes = json.load(f)

# Flip structure
flipped = {}
for field, image_dict in attributes.items():
    for image_id, value in image_dict.items():
        if image_id not in flipped:
            flipped[image_id] = {}
        flipped[image_id][field] = value

df = pd.DataFrame.from_dict(flipped, orient="index").reset_index()
df.rename(columns={"index": "image_id"}, inplace=True)
df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])

# --- Keywords ---
mood_keywords = {
    "Cozy": ["cozy", "warm", "comfortable", "blanket", "fireplace", "pillow", "bed", "relax", "indoors", "candle", "fuzzy", "soft", "snug"],
    "Dreamy": ["dreamy", "fantasy", "cloud", "ethereal", "floating", "fog", "mist", "sparkle", "glow", "illusion"],
    "Romantic": ["romantic", "love", "passion", "valentine", "hearts", "date", "couple", "kiss", "affection", "roses", "bouquet"],
    "Minimalist": ["minimalist", "simple", "clean", "decluttered", "empty", "space", "monochrome", "white", "neat", "order"],
    "Vibrant": ["vibrant", "colorful", "bright", "neon", "pop", "bold", "intense", "saturated", "electric"],
    "Vintage": ["vintage", "old", "retro", "nostalgic", "grainy", "sepia", "classic", "antique", "film"],
    "Natural": ["natural", "outdoors", "nature", "plants", "trees", "forest", "sunlight", "leaves", "wildlife", "flowers", "greenery"],
    "Adventurous": ["adventurous", "explore", "travel", "journey", "climb", "hike", "mountain", "backpack", "camp", "roadtrip"]
}

color_keywords = {
    "Pink": ["pink", "rose", "blush", "rosy", "magenta"],
    "Red": ["red", "crimson", "scarlet", "maroon", "ruby"],
    "Orange": ["orange", "tangerine", "peach", "apricot"],
    "Yellow": ["yellow", "gold", "sunshine", "lemon", "amber"],
    "Green": ["green", "emerald", "forest", "lime", "olive", "mint"],
    "Blue": ["blue", "azure", "navy", "sky", "indigo", "teal"],
    "Purple": ["purple", "lavender", "violet", "plum", "orchid"],
    "White": ["white", "ivory", "pearl", "snow", "milk"]
}

def match_any(tag_list, keywords):
    return any(kw in tag_list for kw in keywords)

# --- Match Images to ALL possible mood-color combos ---
results = []
for _, row in df.iterrows():
    tags = row["tags"]
    for mood, mood_words in mood_keywords.items():
        for color, color_words in color_keywords.items():
            mood_match = match_any(tags, mood_words)
            color_match = match_any(tags, color_words)
            if mood_match or color_match:
                results.append({
                    "image_id": row["image_id"],
                    "tags": tags,
                    "mood": mood if mood_match else None,
                    "color_theme": color if color_match else None
                })

# --- Create DataFrame ---
filtered_df = pd.DataFrame(results)

# --- Save CSV ---
os.makedirs("data", exist_ok=True)
filtered_df.to_csv(OUTPUT_CSV, index=False)

# --- Copy Matched Images ---
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
unique_image_ids = filtered_df["image_id"].unique()
copied = 0
for image_id in unique_image_ids:
    src = os.path.join(IMAGE_ROOT, f"{image_id}.jpg")
    dst = os.path.join(OUTPUT_IMAGE_DIR, f"{image_id}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1

print(f"Copied {copied} unique images to {OUTPUT_IMAGE_DIR}")
print(f"Final labeled rows (with mood/color duplication allowed): {len(filtered_df)}")
print(f"Unique labeled images: {len(unique_image_ids)}")
