# Visualizing Vibes: Abstract Aesthetic Image Generation from Mood and Color Using a Conditional VAE

---

## Overview

**MoodBoard Generator** is an AI-powered image generation tool that creates **abstract aesthetic visuals** based on two simple inputs: a **mood** (e.g., "dreamy", "romantic") and a **color theme** (e.g., "blue", "red"). Instead of producing photorealistic scenes, it generates **blurred, emotional color textures**, like visual vibes, that reflect how a specific mood-color pairing might *feel*.

This project was designed for creators, designers, and moodboard lovers who want to **visually represent an emotional tone** or build aesthetic content foundations.

---

## What This Project Actually Does

This project trains a Conditional Variational Autoencoder (CVAE) to generate **blurred, abstract aesthetic images** from combinations of mood and color labels. It does not aim to generate realistic scenes or objects. Instead, each image represents an emotional impression — a visual "vibe" — derived from ~12,000 aesthetic photographs from Pexels.


### What you get:
- 64x64 RGB image "moodscapes"
- Soft gradients and abstract color compositions
- Unique results for each mood-color combination

---

## What You Can Use This For
* Moodboard builders for designers and creatives
* Branding explorations
* Generative design palette development
* AI x emotion visualizations
* Lightweight aesthetic AI tools

---

## Model Architecture

- **Model Type**: Conditional Variational Autoencoder (CVAE)
- **Conditioning**: Dual (one-hot encoded mood + color)
- **Latent Dimension**: 64
- **Image Size**: 64 x 64 x 3 (RGB)
- **Input Vector**: Concatenation of flattened image and condition vector (12288 + 16)
- **Loss Function**: CVAE loss (reconstruction + KL divergence)
- **Training Epochs**: Up to 200
- **Progress Images**: Generated every 25 epochs during training to show evolution

---

## Folder Structure
```
├── configs/
│   └── config.yaml                     # Model and training configuration
├── data/
│   ├── filtered_images/               # Preprocessed image patches
│   ├── filtered_mood_color_dataset.csv # Image labels (mood + color)
├── model/
│   └── cvae.py                         # CVAE architecture
├── outputs/
│   ├── checkpoints/                    # Saved trained models (.pth)
│   └── generated/                      # Generated image outputs
├── scripts/
│   ├── train_cvae.py                   # CVAE training script
│   └── generate.py                     # Image generation script (from terminal)
├── utils/
│   ├── dataset.py                      # Custom dataset class
│   └── train_utils.py                  # Losses, plots, and saving
├── gui.py                              # GUI definition
├── run_gui.sh                          # Launch script for GUI
├── setup_env.sh                        # Conda environment setup
├── train_cvae.sh                       # SLURM script to train (optional)
├── requirements.txt                    # All required Python packages
```

---

## Accessing and Running on Quest
* Users will not need to retrain the model. All evaluation will be done using the pre-trained model checkpoint and GUI.

### 1. Clone the Repo into Quest
```bash
git clone https://github.com/hanna8008/aesthetic_moodboard_generation.git
cd aesthetic_moodboard_generation
```

### 2. Log into Quest
```bash
ssh -X your_netid@login.quest.northwestern.edu
```

>'-X' enables GUI support 

### 3. Setup the Conda Envrionment (First Time Only)
```bash
bash setup_env.sh
```

This will:
* Create a Conda envrionment called 'moodgen'
* Install all required packages from 'requirements.txt'

To activate manually later:
```bash
conda activate moodgen
```

### 4. Run the GUI
```bash
bash run_gui.sh
```

This will:
* Activate the 'moodgen' envrionment
* Launch 'gui.py' with dropdown menus for mood and color
* Display the generated image in a pop-up window

> Make sure you're on a login node with GUI support and have used 'ssh -X'

---

## Extra Criteria - GUI Overview

The GUI is built using [Gradio](https://www.gradio.app/) and allows users to interactively generate aesthetic images based on mood and color labels.

### Features:
- **Dropdown Menus**: Select one of eight moods and eight color themes
- **Generate Button**: Calls the backend script (`scripts/generate.py`) with the selected mood and color
- **Output Preview**: Displays the generated image in real-time
- **Gradio UI**: Lightweight, web-based interface with soft, aesthetic styling

### How It Works:
1. User selects a mood (e.g., `dreamy`) and a color (e.g., `blue`)
2. Gradio calls the function `generate_and_return_image()`, which runs:
   ```bash
   python scripts/generate.py --mood dreamy --color blue
   ```

---

## Future Improvements
* Support for caption-based conditioning (e.g., "lavender sunset")
* Higher resolution outputs (128x128 or 256x256)
* Diffusion model variant

---

## Additional Notes:
* Trained model is saved in 'outputs/checkpoints/cvae.pth'
* GUI output images are saved in 'outputs/generated/'
* If display issues arise, ensure your terminal supports X11

---

## Image Source: ##
### Title: Pexels 110k 512p JPEG ###
https://www.kaggle.com/datasets/innominate817/pexels-110k-512p-min-jpg

---

## References and Tools Used:
1. [Pexels 110k Dataset (Kaggle)](https://www.kaggle.com/datasets/innominate817/pexels-110k-512p-min-jpg)
2. [CVAE baseline structure](https://github.com/unnir/cVAE)
3. Northwestern’s EECS Autoencoders Lecture by Dr. D'Arcy
4. [PyTorch Dataset Dataloader Guide (Kaggle)](https://www.kaggle.com/code/pinocookie/pytorch-dataset-and-dataloader)
5. [Conditional VAE From Scratch - Medium](https://medium.com/@sofeikov/implementing-conditional-variational-auto-encoders-cvae-from-scratch-29fcbb8cb08f)