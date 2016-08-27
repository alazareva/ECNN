from ecnn.defaults import *


def get_filter_size(real_shape, output_shape=[32,32,3], square=True):
    return (4, 4)

def get_number_of_filters():  # for now returns squared filters but
    return 6

def get_desnse_layer_size():
    return 100

def get_remove_index(number_of_layers):
    assert number_of_layers > 0
    return number_of_layers - 1

def iterations(number_of_training_parameters):
    return 100  # could change this to be based on params

def learning_rate(number_of_training_parameters):
    return LEARNING_RATE

def batch_size():
    return BATCH_SIZE

def selection_function(model_summary):
    return model_summary.validation_x_entropy