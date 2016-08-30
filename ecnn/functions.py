from ecnn.defaults import *
import numpy as np


# TODO these should be functions that maximize the target value


def get_filter_size(output_shape, square=True):
    height, width, _ = output_shape
    min_height = max(int(height / 20), MIN_FILTER_SIZE)
    max_height = min(int(height / 4), MAX_FILTER_SIZE)
    height = np.random.randint(min_height, max_height)
    return (height, height)

def get_number_of_filters():  # for now returns squared filters but
    return np.random.randint(MIN_FILTERS, MAX_FILTERS)

def is_max_pooling(output_shape):
    height, width, _ = output_shape
    if height >= 10 and width >= 10:
        choice = np.random.randint(0,100)
        if choice < 30:
            return True
    return False




def get_desnse_layer_size():
    return np.random.randint(MIN_DENSE_LAYER_SIZE, MAX_DENSE_LAYER_SIZE)

def get_remove_index(number_of_layers):
    assert number_of_layers > 0
    return number_of_layers - 1

def iterations(number_of_training_parameters):
    return 50  # could change this to be based on params

def learning_rate(number_of_training_parameters):
    return LEARNING_RATE

def batch_size():
    return BATCH_SIZE

def keep_prob_conv():
    return 0.9

def keep_prob_dense():
    return 0.7

selection_function = lambda model: model.validation_accuracy

def regularization():
    return 0.004

def stopping_rule():
    loss = float('inf')
    def c(new_loss):
        nonlocal loss
        old_loss, loss = loss, new_loss
        return  np.isclose(loss, old_loss)
    return c

