import tensorflow as tf
from model import *
from train_config import *


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
    pass