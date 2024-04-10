import tensorflow as tf

###CONSTANTS
DATA_PATH = "data/flickr30k_images/"
IMAGES_PATH = "data/flickr30k_images/flickr30k_images/"
IMAGE_SIZE=(224, 224)
VAL_FRACTION=0.05
SEQ_LENGTH=80
VOCAB_SIZE=10000
BATCH_SIZE=64
EPOCHS=20
AUTOTUNE=tf.data.AUTOTUNE

#preprocessing

STRIP_CHARS = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"

###
