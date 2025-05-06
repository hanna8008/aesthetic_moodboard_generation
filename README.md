# CVAE Aesthetic Image Generator

## Project Features
* Conditional Variational Autoencoder (CVAE) trained on filtered aesthetic image dataset
* One-hot encoding of 8 aesthetic mood classes (cozy, dreamy, romantic, minimlaist, vibrant, )
* TBD TO BE INSERTED: dual conditioning on mood and color via one-hot encoding
*  Real-time image generation using a Graphical User INterface (GUI)
* Modularized code with support for future retraiing or dataset expansion

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

## Acessing and Running on Quest
* Users will not need to retrain the model. All evaluation will be done using the pre-trained model checkpoint and GUI.

### 1. Log into Quest
```bash
ssh -X your_netid@login.quest.northwetsern.edu
```

>'-X' is required to forward the GUI (X11).

### 2. Setup the Conda Envrionment (First Time Only)
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

### 3. Run the GUI
```bash
bash run_gui.sh
```

This will:
* Activate the 'moodgen' envrionment
* Launch 'gui.py' with dropdown menus for moood and color
* Display the genereated image in a pop-up window

> Make sure you're on a login node with GUI support and have used 'ssh -X'


## Notes
* Trained model is saved in 'outputs/checkpoints/cvae.pth'
* GUI output images are saved in 'outputs/generated/'
* If display issues arise, ensure your terminal supports X11



## Image Source: ##
### Title: Pexels 110k 512p JPEG ###
https://www.kaggle.com/datasets/innominate817/pexels-110k-512p-min-jpg

## Model architecture(s): ##
For this project, I will create a Conditional Variational Autoencoder (CVAE) and train it from scratch. Compared to a basic VAE, the CVAE will allow me to generate new images based on a provided label or condition (i.e. mood category ).

## Extra Criteria: ##  
I plan to implement a Gallery GUI that allows users to control the aesthetic of the generated image by giving both a list of mood categories to select from and filtering visual attributes (e.g., warmth, saturation, brightness) with a sliding adjustment feature. The GUI will display a grid of AI-generated images matching the mood and visual attributes, helping users to explore and compare results for creative inspiration or design use.


## Sources I used to Help Me:
1. Data Set: https://www.kaggle.com/datasets/innominate817/pexels-110k-512p-min-jpg
2. CVAE: https://github.com/unnir/cVAE
3. Dr. D'Arcy's L.03 Autoencoders Presentation
4. https://www.kaggle.com/code/pinocookie/pytorch-dataset-and-dataloader
5. https://medium.com/@sofeikov/implementing-conditional-variational-auto-encoders-cvae-from-scratch-29fcbb8cb08f
6. https://github.com/unnir/cVAE/blob/master/cvae.py


## Setup Envrionment
### 1. Clone the repo & move into it
```bash
git clone https://github.com/hanna8008/aesthetic_moodboard_generation.git
cd aesthetic_moodboard_generation
```


## Next Steps
* Add data to improve generated photo
* Genereate the gui
* Update the images for moods and colors for the 8 of each that I chose
* Update README to inform how to run the envrionment I created for them to run on Quest
