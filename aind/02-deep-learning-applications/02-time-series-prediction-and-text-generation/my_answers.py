import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    i = 0
    while i < len(series) - window_size:
        X.append(series[i:i + window_size])
        y.append(series[i + window_size])
        i += 1

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y


# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1), activation='tanh'))
    model.add(Dense(1))
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # working with text input, we can work
    # with string constants like string.ascii_... or string.digits 
    # and make a workaround, depending on which characters we want to 
    # exclude 

    unique_characters = []
    for character in text:
        if character not in string.ascii_lowercase and \
        character not in string.ascii_uppercase and \
        character not in punctuation:               # string.ascii_uppercase can be left out, because we transformed the text already
            unique_characters.append(character)
    for unique_character in set(unique_characters):
        text = text.replace(unique_character, ' ')
    return text


### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = 0
    while i < len(text) - window_size:
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size
        
    return inputs,outputs


# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model