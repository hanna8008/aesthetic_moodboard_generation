#gui.py

# -------------------------------------------------------
# gui.py
# -------------------------------------------------------

# --- Import ---
import gradio as gr
import subprocess 
import os
from PIL import Image


# --- Configurable Tags ---
MOOD_OPTIONS = ['cozy', 'dreamy', 'romantic', 'minimalist', 'vibrant', 'vintage', 'natural', 'adventurous']
COLOR_OPTIONS = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'white']
OUTPUT_PATH = "outputs/generated"


def generate_and_return_image(mood, color):
    try:
        subprocess.run(
            ["python", "scripts/generate.py", "--mood", mood, "--color", color],
            check=True
        )
        image_path = os.path.join(OUTPUT_PATH, f"{mood}_{color}.png")
        if os.path.exists(image_path):
            return image_path
        else: 
            return f"Image not found: {image_path}"
    except subprocess.CalledProcessError as e:
        return f"Generation failed: {str(e)}"


# --- Launch Garadio ---
"""custom_theme = gr.themes.Base(
    primary_hue = "pink",
    secondary_hue = "rose",
    font=["Helvetica,", "sans-serif"]
).set(
    #soft cream
    body_background_fill="#fffaf6",
    #pastel pink
    button_primary_background_fill="#fcd6e5",
    #warm coca
    button_primary_text_color="#5c4a4a",
    #match main bg
    block_background_fill="#fffaf6",
    #warm coca
    text_color_primary="#5c4a4a",
    #dusty rose
    border_color_primary="#e7bfcf"
)"""


with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="rose")) as demo:
    gr.Markdown(
        "<h1 style='text-align: center; color: #5c4a4a;'> Aesthetic Image Generator </h1>"
        "<p style='text-align: center; color: #5c4a4a;'> Create Art with Your Mood + Color </p>"
    )

    with gr.Row():
        mood_dropdown = gr.Dropdown(label="Mood", choices=MOOD_OPTIONS)
        color_dropdown = gr.Dropdown(label="Color", choices=COLOR_OPTIONS)
    
    generate_button = gr.Button("Generate Image")
    output_image = gr.Image(
        type="numpy", 
        label="Your Generated Art",
        height=384,
        width=384,
        show_label=True)

    generate_button.click(
        fn=generate_and_return_image,
        inputs=[mood_dropdown, color_dropdown],
        outputs=output_image
    )


demo.launch()
