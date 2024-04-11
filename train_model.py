#!/usr/bin/python3

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras import layers

from train_config import *
import argparse
from data_processing import build_tokenizer, build_image_augmenter,  decode_and_resize
from model import TransformerDecoderBlock, TransformerEncoderBlock, ImageCaptioningModel, get_cnn_model
import keras


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

##THIS FILE gonna train models with specified params

def parse_args():
    parser = argparse.ArgumentParser(description="Set hyperparameters for the model training.")

    # Adding the hyperparameter arguments with their default values
    parser.add_argument('--seq_length', type=int, default=32,
                        help='Input sequence length')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--embed_dim', type=int, default=1028,
                        help='Dimensionality of the embedding layer')
    parser.add_argument('--ff_dim', type=int, default=256,
                        help='Dimensionality of the feedforward network model')
    parser.add_argument('--enc_heads', type=int, default=2,
                        help='Number of heads in the encoder multi-head attention mechanism')
    parser.add_argument('--dec_heads', type=int, default=4,
                        help='Number of heads in the decoder multi-head attention mechanism')
    parser.add_argument('--artifact_dir', type=str, default="./assets",
                        help='Directory to save artifacts')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for training')

    args = parser.parse_args()
    return args

args = parse_args()

# Assigning values from args directly
SEQ_LENGTH = args.seq_length
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
EMBED_DIM = args.embed_dim
FF_DIM = args.ff_dim
ENC_HEADS = args.enc_heads
DEC_HEADS = args.dec_heads
ARTIFACT_DIR = args.artifact_dir
LR = args.lr


print("EPOCHS", EPOCHS)
print("embed_dim ", EMBED_DIM)


###Creating datasets
captionings_df = pd.read_csv(os.path.join(DATA_PATH, "results.csv"), sep="|").dropna()
captionings_df.columns = ["image_name", "comment_number", "comment"]
captionings_df["image_name"] = IMAGES_PATH + "/" + captionings_df["image_name"] 

#ADDING START AND END special tokens
captionings_df["comment"] = "<START> " + captionings_df["comment"] + " <END>"
captionings_df = captionings_df.sample(frac=1,
                                       random_state=42,
                                       replace=False,
                                       )

n_train_examples = int(len(captionings_df) * (1 - VAL_FRACTION))

train_captionings_df = captionings_df[ : n_train_examples]
val_captionings_df = captionings_df[n_train_examples : ]

print("Train image-text examples: ", train_captionings_df.shape[0])
print("Validation image-text examples: ", val_captionings_df.shape[0])


##Prepare tokinzer
tokenizer = build_tokenizer()
tokenizer.adapt(train_captionings_df["comment"].tolist())
print("Tokenizer is ready")


#Create TF-datasets
def process_input(img_path, captions):
    return decode_and_resize(img_path), tf.reshape(tokenizer(captions), shape=(1, SEQ_LENGTH))

def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset

train_dataset = make_dataset(train_captionings_df["image_name"].tolist(),
                             train_captionings_df["comment"].tolist())

val_dataset = make_dataset(train_captionings_df["image_name"].tolist(),
                             train_captionings_df["comment"].tolist())

print("TF-Datasets are created")



#Building model
base_model = keras.applications.efficientnet.EfficientNetB1(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

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
