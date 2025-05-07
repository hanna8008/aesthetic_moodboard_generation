import os
from PIL import Image
import matplotlib.pyplot as plt

# --- Path and Settings ---
progress_dir = "../outputs/generated"
output_image_name = "training_progress_dreamy_blue.png"
epochs = list(range(25, 201, 25))  # [25, 50, ..., 200]

images = []
labels = []

for epoch in epochs:
    filename = f"progress_cvae_moodboard_generator_e{epoch}_dreamy_blue.png"
    img_path = os.path.join(progress_dir, filename)

    if os.path.exists(img_path):
        print(f"Found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        images.append(img)
        labels.append(f"Epoch {epoch}")
    else:
        print(f"⚠️ Missing: {img_path}")

if not images:
    print("No progress images found!")
    exit()

# --- Plot All Images Side by Side with Titles ---
fig, axs = plt.subplots(1, len(images), figsize=(3 * len(images), 3))

for i, (img, label) in enumerate(zip(images, labels)):
    axs[i].imshow(img)
    axs[i].axis("off")
    axs[i].set_title(label, fontsize=8, pad=6)

# Add a suptitle above all subplots
plt.suptitle("CVAE Training Progress: dreamy + blue", fontsize=14, y=1.05)

plt.tight_layout()
save_path = os.path.join(progress_dir, output_image_name)
plt.savefig(save_path, bbox_inches='tight')
plt.show()

print(f"\nProgress visualization saved to: {save_path}")
