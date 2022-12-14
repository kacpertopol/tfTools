#!/usr/bin/env python

# SET LOGGING IN TENSOR FLOW

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

# IMPORT MODULES

import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy

import tfvis

# CONSTANTS

## EPOCHS
MX_EPOCHS = 100

## NUMBER OF DEEP LAYERS
DEEP_LAYERS = 5

## NUMBER OF NEURONs FOR DEEP LAYERS
DEEP_NEURONS = 50

## NUMBER OF INPUTS
INPUT_NUMBER = 3

## NUMBER OF OUTPUTS
OUTPUT_NUMBER = 2

## SIZE OF THE TRAINING SET
TR_SIZE = 10000

## SIZE OF THE VALIDATION SET
VL_SIZE = 2000

# STATIC MODEL

init_uniform = keras.initializers.RandomUniform(minval = -1.0 , maxval = 1.0 , seed = 42)

model_layers = []

l_in = keras.layers.Input(shape = [INPUT_NUMBER])

#l_in = keras.layers.Dense(1 , input_shape = [1] , 
#        kernel_initializer = init_uniform ,
#        bias_initializer = init_uniform)

model_layers.append(l_in)

for deep_layer in range(DEEP_LAYERS):

    l_hidden = keras.layers.Dense(DEEP_NEURONS , activation = "selu" , 
            kernel_initializer = init_uniform ,
            bias_initializer = init_uniform)

    model_layers.append(l_hidden)

l_out = keras.layers.Dense(OUTPUT_NUMBER , 
        kernel_initializer = init_uniform ,
        bias_initializer = init_uniform)

model_layers.append(l_out)

model = keras.models.Sequential(model_layers)

print("STATIC MODEL SUMMARY:")

model.summary()

# SETTING UP THE TRANING, VALIDATION AND TEST SETS

x_train = np.random.normal(loc = 0.0 , scale = 1.0 , size = (TR_SIZE , INPUT_NUMBER))
x_valid = np.random.normal(loc = 0.0 , scale = 1.0 , size = (VL_SIZE , INPUT_NUMBER))

y_train = model(x_train)
y_valid = model(x_valid)

# MODEL FOR TRAINING

init_train = keras.initializers.LecunNormal()

model_layers_train = []

l_in = keras.layers.Input(shape = [INPUT_NUMBER])

model_layers_train.append(l_in)

for deep_layer in range(DEEP_LAYERS):

    l_hidden = keras.layers.Dense(DEEP_NEURONS , activation = "selu" , 
            kernel_initializer = init_train ,
            bias_initializer = init_train)

    model_layers_train.append(l_hidden)

l_out = keras.layers.Dense(OUTPUT_NUMBER , 
        kernel_initializer = init_train ,
        bias_initializer = init_train)

model_layers_train.append(l_out)

model_train = keras.models.Sequential(model_layers_train)

print("TRAINING MODEL SUMMARY:")

model_train.summary()

opt_train = keras.optimizers.Adam(learning_rate = 0.001 , beta_1 = 0.9 , beta_2 = 0.999)

model_train.compile(loss = "mean_squared_error" , optimizer = opt_train)

tfvis.save_visualization("model_example_tfvis" , model_train , 
                            first_iteration = True ,
                            last_iteration = False ,
                            transpose = True ,
                            addActivation = True)

print("# model summary")

print("max expochs : " + str(MX_EPOCHS))

for i in range(MX_EPOCHS):
    print("# epoch : " + str(i))

    history = model_train.fit(
                x_train , y_train , 
                epochs = 1 ,
                validation_data = (x_valid , y_valid))

    print(str(history.params))

    print(str(history.history))

    ni = i + 2
    if ni >= MX_EPOCHS:
        ni = None

    tfvis.save_visualization("model_example_tfvis" , model_train ,
                                iteration = i + 1 ,
                                first_iteration = False ,
                                last_iteration = (i == MX_EPOCHS - 1) ,
                                transpose = True , 
                                otherinfo = history.history,
                                addActivation = True)

