import tensorflow as tf
from model import *
from train_config import *
import pandas as pd
import datetime
import os
import json

def load_trained_model_weights(path_to_weights):
    base_model = keras.applications.efficientnet.EfficientNetB1(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    #Init new models with same configuration
    cnn = get_cnn_model(base_model)

    encoder = TransformerEncoderBlock(
        embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=ENC_HEADS
    )
    decoder = TransformerDecoderBlock(
        embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=DEC_HEADS, 
    )

    caption_model = ImageCaptioningModel(
        cnn_model=cnn,
        encoder=encoder, 
        decoder=decoder
    )

    #Necessary steps to init model
    #to be able to load saved weights 
    cnn_input = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    training = False
    decoder_input = tf.keras.layers.Input(shape=(None,))
    caption_model([cnn_input,  training, decoder_input])

    #loading weights
    caption_model.load_weights(path_to_weights)

    return caption_model


def save_training_history(history, file):
    
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    train_accuracy = history.history["loss"]
    val_accuracy = history.history["loss"]

    #create pandas df
    history_df = pd.DataFrame({"loss" : train_loss,
                                "accuracy" : train_accuracy,
                                "val_loss" : val_loss,
                                 "val_accuracy" : val_accuracy})
    
    history_df.to_csv(file, index=False)
    print("Training history saved to: ", file)


def save_trial_config(current_config):
    # Extract the directory path from the configuration
    artifact_dir = current_config.get("ARTIFACT_DIR")
    
    if not artifact_dir:
        raise ValueError("ARTIFACT_DIR is not set in the configuration")

    # Create the directory if it does not exist
    os.makedirs(artifact_dir, exist_ok=True)  # `exist_ok=True` avoids an error if the directory already exists

    # Define the file path for the trial configuration JSON file
    trial_config_path = os.path.join(artifact_dir, "trial_config.json")

    # Write the configuration dictionary to a JSON file
    with open(trial_config_path, "w") as file:
        json.dump(current_config, file, indent=6)

    print("Trial config saved: ", trial_config_path)