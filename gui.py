# -------------------------------------------------------
# GUI Code
# -------------------------------------------------------
# Launches an interactive web GUI using Gradio to let users
# generate aesthetic images from mood + color selections.
#
# On user input, the GUI triggers the backend generation script
# and displays the resulting image. Built for ease of use and
# aesthetic appeal with themed styling and dropdown selections.



# --- Import ---
import gradio as gr
import subprocess 
import os
from PIL import Image



# --- Configuration: Mood and Color Options ---
#list of mood labels shown in the dropdown menu
MOOD_OPTIONS = ['cozy', 'dreamy', 'romantic', 'minimalist', 'vibrant', 'vintage', 'natural', 'adventurous']

#list of color labels shown in the dropdown menu
COLOR_OPTIONS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white']

#directory where generated images are saved by generate.py
OUTPUT_PATH = "outputs/generated"




# --- Trigger Image Generation ---
#This function runs the image generation script using subprocess,
# then checks and returns the resulting image path (or an error message)
def generate_and_return_image(mood, color):
    try:
        #run the generate.py script with mood and color passed in as arguments
        subprocess.run(
            ["python", "scripts/generate.py", "--mood", mood, "--color", color],
            #raises an error if the script fails
            check=True
        )

        #construct the expected path to the output image
        image_path = os.path.join(OUTPUT_PATH, f"generated_{mood}_{color}.png")

        #if the file was created, return its path (to be displayed)
        if os.path.exists(image_path):
            return image_path
        else: 
            #if no file found, return an error message
            return f"Image not found: {image_path}"

    #if the script itself throws an error (e.g., crash), catch and return message
    except subprocess.CalledProcessError as e:
        return f"Generation failed: {str(e)}"



# --- Gradio GUI Interface ---
#create the Gradio interface using a themed layout
#using the built-in 'Soft' theme with pink and rose hues
with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="rose")) as demo:
    #display a centered title and subtitleusing HTML + Markdown
    gr.Markdown(
        "<h1 style='text-align: center; color: #5c4a4a;'> Abstract Aesthetic Image Generator </h1>"
        "<p style='text-align: center; color: #5c4a4a;'> Create Art based on Mood + Color </p>"
    )

    #layout: row for two dropdowns side by side
    with gr.Row():
        #dropdown to select mood
        mood_dropdown = gr.Dropdown(label="Mood", choices=MOOD_OPTIONS)
        #dropdown to select color
        color_dropdown = gr.Dropdown(label="Color", choices=COLOR_OPTIONS)
    
    #create a button the user clicks to trigger image generation
    generate_button = gr.Button("Generate Image")
    
    #create an output image component to display the result
    output_image = gr.Image(
        #return the filepath from the function
        type="filepath", 
        #label above the image
        label="Your Generated Art",
        #display dize
        height=640,
        width=640,
        show_label=False)



    #define what happens when the button is clicked:
    #it calls 'generate_and_return_image()' with mood and color,
    #and displays the result in the image component
    generate_button.click(
        fn=generate_and_return_image,
        inputs=[mood_dropdown, color_dropdown],
        outputs=output_image
    )

#launch the web interface in a local browser tab
demo.launch(share=True)
