# -- NB: always make sure python version in which model was trained should match local or hugging face environment otherwise model won't work

# -- fix model loading issues when a .pkl file saved on Linux (using PosixPath) is loaded on Windows. 
# -- But it should be done before importing Fastai or loading the model. 
# -- comment the following when uploading to huggingface

# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath 

# ---------------------             original demo code

# import gradio as gr
# import sys
 
# def greet(name):
#     return "Hello " + name + "!!" + "Running Python version:" + sys.version

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()

# --------------------               main code

from fastai.vision.all import *
import gradio as gr

model = load_learner('models/bird-recognizer-model_resnet34.pkl')

bird_categories = [
    "Asian Koel (Kokil)",
    "Black Drongo (Finge)",
    "Brahminy Kite (Shankh Chil)",
    "Common Kingfisher (Machh Ranga)",
    "Common Myna (Shalik)",
    "House Crow (Pati Kak)",
    "House Sparrow (Chorui)",
    "Indian Pond Heron (Kani Bok)",
    "Little Cormorant (Pankowri)",
    "Oriental Magpie-Robin (Doel)",
    "Red-vented Bulbul (Bulbul)",
    "Rock Pigeon (Payerra)",
    "Rose-ringed Parakeet (Tiya Pakhi)",
    "Spotted Dove (Telaghughu)",
    "White-breasted Waterhen (Dahuk)"
]

def recognize_image(image):
    pred, idx, probs = model.predict(image)
    return dict(zip(bird_categories, map(float, probs)))

image = gr.Image()
label = gr.Label()
examples = [
    'test_images/test-1.jpg',
    'test_images/test-2.jpg',
    'test_images/test-3.jpg',
    'test_images/test-4.jpg',
    'test_images/test-5.jpg',
    'test_images/test-6.jpg',
    'test_images/test-7.jpg',
    'test_images/test-8.jpg',
    'test_images/test-9.jpg',
    'test_images/test-10.jpg',
    'test_images/test-11.jpg',
    'test_images/test-12.jpg',
    'test_images/test-13.jpg',
    'test_images/test-14.jpg',
    'test_images/test-15.jpg'
    ]


iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)

iface.launch(inline=False)