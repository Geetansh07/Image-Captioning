import pickle
import random

import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


print("Let's start")
print("--------------------------------------------------------------------------------------------------------------")
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

filename = "captions.txt" # file that contains all the captions for all 8000 images

print("All captions Imported")
print("For each image there are 5 captions")
print("------------------------------------------------------------------------------------------------------")

doc = load_doc(filename) # loading the captions in a doc variable

#Example of what doc contains
print("Examples of a what a document contains: ",doc[:500])

print("----------------------------------------------------------------------------------------------------------")

# making a dictionary so that we can separate all image name and their captions.
def load_descriptions(doc):
    mapping = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        if len(line) < 2:
            continue
        # take the first token as the image id, the rest as the description
        image_id, image_desc = tokens[0], tokens[1:]
        # extract filename from image id
        image_id = image_id.split('.')[0]
        # convert description tokens back to string
        image_desc = ' '.join(image_desc)
        # create the list if needed
        if image_id not in mapping:
            mapping[image_id] = list()
        # store description
        mapping[image_id].append(image_desc)
    return mapping


# making a descriptions dict
descriptions = load_descriptions(doc)

print("Done separating all the image name and their captions")
print("------------------------------------------------------------------------------------------------------------")
print('Length of descriptions dictionary : %d ' % len(descriptions))





