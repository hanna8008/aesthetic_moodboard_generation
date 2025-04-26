#this is a script to organize the Pexel images by mood tags and resize them for training 

import os
import json 
from PIL import Image
import shutil
import kagglehub

# Download latest version
path = kagglehub.dataset_download("innominate817/pexels-110k-512p-min-jpg")

print("Path to dataset files:", path)



#Settings
original_image_directory = "imageGenerationProject/data/moods/all_images"
metadata_file = "/Users/hannazelis/.cache/kagglehub/datasets/innominate817/pexels-110k-512p-min-jpg/versions/4/pexels-110k-512p-min-jpg/"
# under this directory includes: attributes_df.json, images folder, pexels-prompts-pairs.json, and tags.txt; want to move this to data folder under imageGenerationProject
output_base_directory = "imageGenerationProject/data/sorted"
resize_to= (128, 128)



#define keywords for each mood
mood_keywords = {
    "Cozy": ["cozy", "warm", "comfortable", "blanket", "fireplace", "pillow", "bed", "relax"],
    "Dreamy": ["dreamy", "fantasy", "whimsical", "clouds", "soft", "dream", "imagination"],
    "Romantic": ["romantic", "love", "passion", "couple", "flowers", "candlelight", "intimate", "roses", "valentine"],
    "Minimalist": ["minimalist", "simple", "clean", "white", "empty", "spacious", "bare"],
    "Vibrant": ["vibrant", "colorful", "bright", "bold", "saturated", "lively", "energetic"],
    "Natural": ["natural", "outdoors", "nature", "forest", "mountain", "river", "wildlife", "green"],
}



#load metadata
print("Loading metadata")
with open(metadata_file, 'r') as f:
    metadata = json.load(f)



#sort and resize
print("Sorting and resizing images")
os.makedirs(output_base_directory, exist_ok=True)
counts = {mood: 0 for mood in mood_keywords.keys()}

for entry in metadata:
    filename = entry.get("filename")
    tags = entry.get("tags", [])
    src_path = os.path.join(original_image_directory, filename)

    if not os.path.exists(src_path):
        print(f"File not found: {src_path}")
        continue

    #check whcih mood this image fits into
    for mood, keywords in mood_keywords.items():
        if any(kw in tags for kw in keywords):
            dest_dir = os.path.join(output_base_directory, mood)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)

            try:
                img = Image.open(src_path).convert("RGB")
                img = img.resize(resize_to)
                img.save(dest_path)
                counts[mood] += 1

            except Exception as e:
                print(f"Error processing {filename}: {e}")
            
            break   #assign to only one mood



#end of script
print("Sorting and resizing complete. Image counts:")
for mood, count in counts.items():
    print(f"{mood}: {count} images")