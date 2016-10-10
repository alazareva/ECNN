from ecnn.defaults import *
import numpy as np
from functools import partial




def iterations():
    return 5


def learning_rate(lr=LEARNING_RATE):
    decrease_rate = 0.75
    lr = lr
    window = []
    window_size = 5
    def f(loss = float('inf')):
        nonlocal window
        nonlocal lr
        nonlocal window_size
        window.append(loss)
        if len(window) == window_size:
            diffs = np.ediff1d(window)
            if np.all(abs(diffs) > np.array(window[:-1])*0.05) and np.mean(diffs > 0) >= 0.5: # if large loss
                # fluctuations
                print("fluctuating", window)
                lr *= decrease_rate
                window = []
            elif np.all(abs(diffs) < np.array(window[:-1])*0.01) and np.all(diffs < 0): # if decreased by
                # small amount
                print("too slow", window)
                lr *= 1/decrease_rate
                window = []
            else:
                window.pop(0)
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
    attrs = [('validation_accuracy', 1), ('trainable_parameters', -0.5)]
    return get_partial_attr(models, attrs)

def sort_on_accuracy(models): #first is the parameter and the second is contribution
    attrs = [('validation_accuracy', 1)]
    return get_partial_attr(models, attrs)


def regularization():
    return 0.004

def stopping_rule():
    window = []
    window_size = 5
    def c(val_acc):
        nonlocal window
        nonlocal window_size
        print('acc', val_acc)
        window.append(val_acc)
        if len(window) == window_size:
            diffs = np.ediff1d(window)
            if np.all(diffs < 0):
                return True
            window.pop(0)
        return False
    return c

