import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras import layers
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io
from PIL import Image
import json
import tensorflow as tf
import numpy as np
import fastapi
from training_utils import *
from data_processing import *

##simple ML API for image captioning 

app = fastapi.FastAPI()

if not os.path.exists('tmp'):
    os.makedirs('tmp')

with open("model_config.json", "r") as conf_f:
    CONFIG = json.load(conf_f)

tokenizer = load_tokenizer("tokenizer.keras")
caption_model = load_trained_model_weights("L_size_model_weights.h5", CONFIG)
vocab = tokenizer.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))
max_decoded_sentence_length = CONFIG["SEQ_LENGTH"] - 1

def generate_caption(path_to_img):
    # Select a random image from the validation dataset
    sample_img = path_to_img

    # Read the image from the disk
    sample_img = decode_and_resize(sample_img, CONFIG["IMAGE_SIZE"])
    img = sample_img.numpy().clip(0, 255).astype(np.uint8)

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = tokenizer([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "end":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace("end", "").strip()
    print("Predicted Caption: ", decoded_caption)

    return decoded_caption

print("SUCCESS")

print(generate_caption("example.jpg"))

# @app.post("/caption")
# async def predict(file: UploadFile = File(...)):
#     if file.content_type.startswith('image/'):

#         # Save the image to the tmp/ folder
#         file_path = f"tmp/{file.filename}"
#         with open(file_path, "wb") as buffer:
#             buffer.write(await file.read())
        