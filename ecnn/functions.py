from ecnn.defaults import *
import numpy as np
from functools import partial



# TODO these should be functions that maximize the target value

'''
def get_filter_size(output_shape, square=True):
    height, width, _ = output_shape
    min_height = min(max(int(height / 20), MIN_FILTER_SIZE), MAX_FILTER_SIZE) #3
    max_height = max(min(int(height / 4), MAX_FILTER_SIZE),MIN_FILTER_SIZE) #8
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


# maybe have a function factory

def get_desnse_layer_size():
    return np.random.randint(MIN_DENSE_LAYER_SIZE, MAX_DENSE_LAYER_SIZE)

def get_remove_index(number_of_layers):
    assert number_of_layers > 0
    return number_of_layers - 1

'''

def iterations(number_of_training_parameters):
    return 50  # could change this to be based on params

'''
def learning_rate(number_of_training_parameters):
    return LEARNING_RATE
'''

def learning_rate(decrease_rate):
    loss = float('inf')
    decrease_rate = decrease_rate
    lr = LEARNING_RATE
    number_of_loss_increases = 0
    number_of_allowed_stagn = 3
    allowed_increases = 3
    number_of_stagn = 0
    def f(new_loss):
        nonlocal loss
        nonlocal lr
        nonlocal number_of_loss_increases
        nonlocal allowed_increases
        nonlocal number_of_stagn
        nonlocal number_of_allowed_stagn
        if new_loss:
            old_loss, loss = loss, new_loss
            print('oldloss , newloss', old_loss, loss)
            if old_loss < loss:
                number_of_loss_increases +=1
                if number_of_loss_increases == allowed_increases:
                    lr *= decrease_rate
            else:
                number_of_loss_increases = 0
                if old_loss-loss <  old_loss*0.005:
                    number_of_stagn +=1
                    if number_of_stagn == number_of_allowed_stagn:
                        lr *= 1/decrease_rate
                else:
                    number_of_stagn = 0
        print('returning lr', lr)
        return lr
    return f





def batch_size():
    return BATCH_SIZE

def keep_prob_conv():
    return 0.8 #0.9

def keep_prob_dense():
    return 0.7

selection_function_val_accuracy = lambda model: model.validation_accuracy


def get_mean(models, attr):
    return np.mean([getattr(model, attr) for model in models])

def sort_on_attrs(model, means, attrs):
    return sum([contrib*(getattr(model, attr) - means[attr])/means[attr] for attr, contrib in attrs])

def get_partial_attr(models, attrs):
    means = {attr:get_mean(models, attr) for attr, _ in attrs}
    return partial(sort_on_attrs, means=means, attrs=attrs)

def sort_on_accuracy_params(models): #first is the parameter and the second is contribution
    attrs = [('validation_accuracy', 1), ('trainable_parameters', -1)]
    return get_partial_attr(models, attrs)


def regularization():
    return 0.004

def stopping_rule():
    val_acc = 0
    number_of_decreases = 0
    max_decreases = 4
    def c(new_val_acc):
        nonlocal val_acc
        nonlocal number_of_decreases
        nonlocal max_decreases
        old_val_acc, val_acc = val_acc, new_val_acc
        print('old_acc, new_acc', old_val_acc, val_acc)
        if val_acc < old_val_acc:
            number_of_decreases +=1
        else:
            number_of_decreases = 0
        return number_of_decreases == max_decreases
    return c

