import tensorflow as tf
import numpy as np
from train_config import STRIP_CHARS, VOCAB_SIZE, SEQ_LENGTH, IMAGE_SIZE, BATCH_SIZE, AUTOTUNE
import re
import keras


def text_standrardization(input_str):
    lowercase = tf.strings.lower(input_str)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(STRIP_CHARS), "")

def build_tokenizer():

    vectorization = tf.keras.layers.TextVectorization(
                        max_tokens=VOCAB_SIZE,
                        output_mode="int",
                        output_sequence_length=SEQ_LENGTH,
                        standardize=text_standrardization,
                                                    )
    return vectorization

def build_image_augmenter(rotation_rate=0.2, contrast=0.3):
    image_augmentation = tf.keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(rotation_rate),
        keras.layers.RandomContrast(contrast),
    ]
    )
    return image_augmentation

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = img / 255.0 
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset