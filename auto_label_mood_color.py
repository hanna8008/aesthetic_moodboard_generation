# -------------------------------------------------------
# Create CSV File with All Data Information Plus Mood + Color
# -------------------------------------------------------
# 
# Automatically labels aesthetic images with mood and color themes
# based on keyword matching from tag metadata. Extracts image-tag pairs
# from a JSON file, identifies relevant mood/color keywords, filters
# valid samples, saves a new CSV, and copies matched images
# into a filtered image directory
#
# Useful for preparing a semi-supervised dataset for CVAE training



# --- Imports ---
import os
import json
import pandas as pd
import shutil
from collections import defaultdict



# --- File Paths ---
#path to the JSON file containing image attributes and tags from original dataset
ATTRIBUTES_PATH = "data/pexels-110k-512p-min-jpg/attributes_df.json"

#directory containing original image files
IMAGE_ROOT = "data/pexels-110k-512p-min-jpg/images"

#output CSV file path for filtered, labeled dataset
OUTPUT_CSV = "data/filtered_mood_color_dataset.csv"

#output directory to store only filtered, matched images
OUTPUT_IMAGE_DIR = "data/filtered_images"



# --- Load JSON ---
#load JSON metadata from file
with open(ATTRIBUTES_PATH, "r") as f:
    attributes = json.load(f)



# --- Flip Structure ---
#reformat structure so each image_id becomes the key, and its tags become values
#create an empty dictionary to hold the reformatted image data
#new format: { image_id: {field1: value, field2: value, ...}}
flipped = {}

#loop through each metadata field in the original JSON (e.g., 'tags', 'width', 'height')
for field, image_dict in attributes.items():
    #within each field, loop through each image and its corresponding value
    for image_id, value in image_dict.items():

        #if this image_id hasn't been added yet
        if image_id not in flipped:
            #initialize an empty dictionary for it
            flipped[image_id] = {}
        
        #add the field and its value to the image_id's entry
        #example: flipped['img_1234']['tags'] = ['cozy']['pink]
        flipped[image_id][field] = value

#convert the flipped dictionary to a Pandas DataFrame
df = pd.DataFrame.from_dict(flipped, orient="index").reset_index()
#rename index to 'image_id
df.rename(columns={"index": "image_id"}, inplace=True)

#ensure 'tags' column always contains a list (not NaN or string)
df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])



# --- Keywords ---
#dictionary of mood categories and their associated keywords
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

#dictionary of color themes and the synonyms or shades
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



# --- Keyword Matching Function ---
#utility: check if *any* keyword from the list exists in the tag list
def match_any(tag_list, keywords):
    return any(kw in tag_list for kw in keywords)



# --- Match Images to ALL possible mood-color combos ---
#match mood + color tags to each image

#hold matched image entries
results = []

#loop through each image row in the DataFrame
for _, row in df.iterrows():

    #extract the list of tags (words associated with the image)
    tags = row["tags"]

    #loop through each mood category and its associated keywords
    for mood, mood_words in mood_keywords.items():
        #loop through each color category and its associated keywords
        for color, color_words in color_keywords.items():

            #check if any of the mood keywordsappear in the image's tag list
            mood_match = match_any(tags, mood_words)
            #check if any of the color keywords appear in the tag list
            color_match = match_any(tags, color_words)

            #if at least one mood OR color is matched
            if mood_match or color_match:

                #append the image's ID and matched mood/color to the results list
                results.append({
                    #ID of the image
                    "image_id": row["image_id"],
                    #original tag list
                    "tags": tags,
                    #assign mood if matched, otherwise None
                    "mood": mood if mood_match else None,
                    #assign color if matched, otherwise None
                    "color_theme": color if color_match else None
                })



# --- Saved Filtered Dataset as CSV ---
#convert the matched results to a DataFrame
filtered_df = pd.DataFrame(results)

#make sure datafolder exists
os.makedirs("data", exist_ok=True)
#save the new filtered dataset
filtered_df.to_csv(OUTPUT_CSV, index=False)



# --- Copy Matching Image Files ---
#create output directory for filtered images if not exists
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

#get unique image IDs from the filtered set
unique_image_ids = filtered_df["image_id"].unique()

#counter for how many images were successfully copied
copied = 0

#loop over unique IDs and copy files from source to filtered folder
#loop through every image ID that matched at least one mood/color combo
for image_id in unique_image_ids:
    
    #build the full path to the original image
    src = os.path.join(IMAGE_ROOT, f"{image_id}.jpg")
    #build the destination path for where the image should be copied
    dst = os.path.join(OUTPUT_IMAGE_DIR, f"{image_id}.jpg")

    #only copy if the source image actually exists (some may be missing)
    if os.path.exists(src):
        #copy the image from its original location to the filtered output folder 
        shutil.copy(src, dst)
        #increment the counter tracking how many images were successfully copied
        copied += 1



# --- Print Final Summary of Labeling Results ---
#output how many unique images were copied into the filtered folder
print(f"Copied {copied} unique images to {OUTPUT_IMAGE_DIR}")

#show how many rows were genereated in the CSV - thsi may include duplicates (same image with different moods/colors)
print(f"Final labeled rows (with mood/color duplication allowed): {len(filtered_df)}")

#show the number of unique image IDs after removing duplicates - each image only counted once here
print(f"Unique labeled images: {len(unique_image_ids)}")
