import tensorflow as tf
import numpy as np
import pandas as pd
import os
from keras import layers

from data_processing import *
from training_utils import *
import fastapi

##simple ML API for image captioning 