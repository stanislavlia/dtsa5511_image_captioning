# Image Captioning model 

## Intro
This repo contains training and inference code for image captioning model (image2text).

An **image captioning model** is an artificial intelligence system designed to automatically generate textual descriptions for images. It integrates elements of computer vision to interpret visual content and natural language processing to formulate coherent sentences. The model typically processes the image through a convolutional neural network to extract visual features, and then uses a recurrent neural network or a transformer-based architecture to generate relevant captions based on those features. This technology is useful in various applications such as aiding visually impaired users, organizing image databases, and enhancing user interactions with multimedia content.
#### Examples
![example_dog](media/caption_examples.png)

#### Application GUI
![example_dog](media/example_dog.png)
![example_dog](media/example_baseball.png)

## Data
In this project, we used publicly available dataset **FlickR30K**.

The **Flickr30k** dataset is a popular benchmark for sentence-based picture portrayal. The dataset is comprised of **31,783** images that capture people engaged in everyday activities and events. Each image has a descriptive caption. Flickr30k is used for understanding the visual media (image) that correspond to a linguistic expression (description of the image). This dataset is commonly used as a standard benchmark for sentence-based image descriptions. 
The size is about **9GB**

Source - https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset

## Model Architecture
#### Architecture diagram
The architecture of our Image captioning model consists of 3 parts:
  1) **CNN** as feature extractor from images. (EfficientNetB1 in our case)
  2) **Transformer Encoder Block** that process image embedding
  3) **Transformer Decoder Block** that generates a caption in autoregressive mode (token by token)

This architecture uses Convolution NN as a feature extractor that produces 2D "embedding" of image. Then, this "embedding" matrix produced by CNN is passed to Transformer Encoder block and then the model starts to generate caption in Transformer Decoder part. The last layer of our model is **softmax** over all words in **vocabulary**.

During the training, we keep our **vocabulary** of constants size - **10 000 words**.

![example_dog](media/architecture.png)

## Training
As I understood that the training procedure was going to be **computationlly intensive and long**, I prepared several python modules to make workflow more clean, maintainable and efficient.
  - Model architecture (model.py)
  - Data preprocessing (data_processing.py)
  - Experiment tracking and saving artifacts (training_utils.py)


### Training script
And the main file in this repo is - **train_model.py** which is a Python script that takes all these modules and train a model with hyperparameters specified in **command line arguments** and at the same time loggin all process both **locally** and to service **Weights & Biases**.

example of usage:
```bash
./train_model.py --embed_dim 1024 --ff_dim 512 --enc_heads 4 --dec_heads 4\
                 --artifact_dir local_runs/model1_artifacts --epochs 40 --lr 0.001
```

This document provides detailed descriptions of the command line arguments available for configuration in the script. These options allow customization of model parameters and training settings.

## Arguments

- `--seq_length`: 
  - **Type**: `int`
  - **Default**: `36`
  - **Description**: Specifies the input sequence length for the model.

- `--batch_size`: 
  - **Type**: `int`
  - **Default**: `128`
  - **Description**: Defines the number of samples to work through before updating the internal model parameters.

- `--epochs`: 
  - **Type**: `int`
  - **Default**: `20`
  - **Description**: Sets the total number of complete passes through the training dataset.

- `--embed_dim`: 
  - **Type**: `int`
  - **Default**: `512`
  - **Description**: Determines the dimensionality of the embedding layer in the neural network.

- `--ff_dim`: 
  - **Type**: `int`
  - **Default**: `256`
  - **Description**: Specifies the dimensionality of the feedforward network model within the transformer.

- `--enc_heads`: 
  - **Type**: `int`
  - **Default**: `2`
  - **Description**: Sets the number of heads in the encoder part of the multi-head attention mechanism.

- `--dec_heads`: 
  - **Type**: `int`
  - **Default**: `4`
  - **Description**: Sets the number of heads in the decoder part of the multi-head attention mechanism.

- `--artifact_dir`: 
  - **Type**: `str`
  - **Default**: `./local_runs/default_run`
  - **Description**: Provides the directory path where training artifacts (like model weights and logs) will be saved.

- `--lr`: 
  - **Type**: `float`
  - **Default**: `0.001`
  - **Description**: Specifies the learning rate used in training the model.


