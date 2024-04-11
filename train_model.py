
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras import layers

from train_config import *
import argparse


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

print("EPOCHS", EPOCHS)
print("embed_dim ", EMBED_DIM)

