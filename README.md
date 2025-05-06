# CVAE Aesthetic Image Generator

## Project Features
* Conditional Variational Autoencoder (CVAE) trained on filtered aesthetic image dataset
* TBD TO BE INSERTED: dual conditioning on mood and color via one-hot encoding
*  Real-time image generation using a Graphical User INterface (GUI)
* Modularized code with support for future retraiing or dataset expansion

## Folder Structure

## Acessing and Running on Quest
* Users will not need to retrain the model. All evaluation will be done using hte pre-trained model checkpoint and GUI .




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
