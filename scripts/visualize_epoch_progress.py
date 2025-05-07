# -------------------------------------------------------
# Visualize Epoch Progress for Dreamy Blue Training Images
# -------------------------------------------------------
# 
# Utility script for visualizing the progression of image geeration
# over training eopchs of a CVAE model.
#
# This script scans the output folder for saved images(generated every N 
# epochs), loads them in order, and displays them in a grid or saves them
# as a single summary image.
#
# Helpful for monitoring model improvements over time and comparing



# --- Imports ---
import os
from PIL import Image
import matplotlib.pyplot as plt



# --- Path and Settings ---
#directory containing the saved progress images from training
progress_dir = "../outputs/generated"

#output file name for the final visualization (grid of images)
output_image_name = "training_progress_dreamy_blue.png"

#define the training epochs you want to visualize progress at 
#example: [25, 50, 75, ..., 200]
epochs = list(range(25, 201, 25))



# --- Load Saved Images ---

#list to hold loaded images for each epoch
images = []
#list to hold corresponding epoch labels for titles
labels = []

#loop through each selected epoch and load the corresponding image
for epoch in epochs:
    #construct the expected filename for the generated image at this epoch
    filename = f"progress_cvae_moodboard_generator_e{epoch}_dreamy_blue.png"
    #full path to image
    img_path = os.path.join(progress_dir, filename)

    #check if hte image file exists before trying to load it
    if os.path.exists(img_path):
        #log success
        print(f"Found: {img_path}")
        #open image and ensure it's in RGB mode
        img = Image.open(img_path).convert("RGB")
        #add image to list
        images.append(img)
        #add label (used for subplot titles)
        labels.append(f"Epoch {epoch}")
    else:
        #warn if expected image is missing
        print(f"⚠️ Missing: {img_path}")

#if no images were found at all, exit the script early
if not images:
    print("No progress images found!")
    exit()



# --- Plot All Images Side by Side with Titles ---
#create a subplot with 1 row and N columns (one for each epoch image)
fig, axs = plt.subplots(1, len(images), figsize=(3 * len(images), 3))

#loop over each loaded image and corresponding label
for i, (img, label) in enumerate(zip(images, labels)):
    #display the image
    axs[i].imshow(img)
    #hide axes/ticks for cleaner display
    axs[i].axis("off")
    #set title as the epoch label
    axs[i].set_title(label, fontsize=8, pad=6)

#add a suptitle above all subplots
plt.suptitle("CVAE Training Progress: dreamy + blue", fontsize=14, y=1.05)

#adjust layout to prevent overlapping titles/subplots
plt.tight_layout()

#construct the full save path for the output summary image
save_path = os.path.join(progress_dir, output_image_name)

#save the plotted figure to disk as a PNG file
plt.savefig(save_path, bbox_inches='tight')

#display it in an interactive window
plt.show()

#final confirmation in terminal
print(f"\nProgress visualization saved to: {save_path}")
