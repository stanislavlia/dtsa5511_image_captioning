import tensorflow as tf
from model import *
import pandas as pd
import datetime
import os
import json
import pickle
import wandb


def load_trained_model_weights(path_to_weights, config):
    base_model = keras.applications.efficientnet.EfficientNetB1(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    #Init new models with same configuration
    cnn = get_cnn_model(base_model)

    encoder = TransformerEncoderBlock(
        embed_dim=config["EMBED_DIM"], dense_dim=config["FF_DIM"], num_heads=config["ENC_HEADS"],
         
    )
    decoder = TransformerDecoderBlock(
        embed_dim=config["EMBED_DIM"], ff_dim=config["FF_DIM"], num_heads=config["DEC_HEADS"], 
        seq_len=config["SEQ_LENGTH"], vocab_size=config["VOCAB_SIZE"]
    )

    caption_model = ImageCaptioningModel(
        cnn_model=cnn,
        encoder=encoder, 
        decoder=decoder
    )

    #Necessary steps to init model
    #to be able to load saved weights 
    cnn_input = tf.keras.layers.Input(shape=(224, 224, 3))
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

    history_df = pd.DataFrame({"loss" : train_loss,
                                "accuracy" : train_accuracy,
                                "val_loss" : val_loss,
                                 "val_accuracy" : val_accuracy})
    
    history_df.to_csv(file, index=False)
    print("Training history saved to: ", file)



def save_trial_config(current_config):
    artifact_dir = current_config.get("ARTIFACT_DIR")
    
    if not artifact_dir:
        raise ValueError("ARTIFACT_DIR is not set in the configuration")

    os.makedirs(artifact_dir, exist_ok=True)
    trial_config_path = os.path.join(artifact_dir, "trial_config.json")

    with open(trial_config_path, "w") as file:
        json.dump(current_config, file, indent=6)

    print("Trial config saved: ", trial_config_path)


def save_tokenizer(tokenizer, path):

    pickle.dump({'config': tokenizer.get_config(),
             'weights': tokenizer.get_weights()}
            , open(path, "wb"))
    
    print("Tokenizer is saved: ", path)




def load_tokenizer(path):
    from_disk = pickle.load(open("tv_layer.pkl", "rb"))
    new_v = tf.keras.preprocessing.TextVectorization.from_config(from_disk['config'])

    # You have to call `adapt` with some dummy data (BUG in Keras)
    new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_v.set_weights(from_disk['weights'])

    return new_v




def log_artifact_to_wandb(run, run_dir, run_name):

    artifact = wandb.Artifact( type="model",
                               name="saved_assets_dir",
                               )
    artifact.add_dir(local_path=run_dir) 
    run.log_artifact(artifact)