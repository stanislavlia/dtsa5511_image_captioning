import tensorflow as tf
import datetime

###CONSTANTS
DATA_PATH = "data/flickr30k_images/"
IMAGES_PATH = "data/flickr30k_images/flickr30k_images/"
IMAGE_SIZE=(224, 224)
VAL_FRACTION=0.05
SEQ_LENGTH=32
VOCAB_SIZE=10000
BATCH_SIZE=64
EPOCHS=20
AUTOTUNE=tf.data.AUTOTUNE

#MODEL HYPERPARAMS
EMBED_DIM=512
FF_DIM=256
ENC_HEADS=1
DEC_HEADS=2


STRIP_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

###

def get_config():
    return {
        "TIMESTAMP" : datetime.datetime.now(),
        "DATA_PATH": DATA_PATH,
        "IMAGES_PATH": IMAGES_PATH,
        "IMAGE_SIZE": IMAGE_SIZE,
        "VAL_FRACTION": VAL_FRACTION,
        "SEQ_LENGTH": SEQ_LENGTH,
        "VOCAB_SIZE": VOCAB_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "AUTOTUNE": AUTOTUNE,
        "STRIP_CHARS": STRIP_CHARS
    }