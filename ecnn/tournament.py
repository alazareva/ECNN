# script to run tourtnament

import os
import pickle
import random
import itertools
import functools
import numpy as np
import copy


from ecnn.tensorflow_model import TensorflowModel
from ecnn.class_defs import *


# TODO maybe use flask for display?
# TODO https://gist.github.com/Mistobaan/dd32287eeb6859c6668d GPU on mac
# TODO use coverage testing
# TODO decouple weights and trained params
# TODO maybe refactor mutations into their own classes, so instead of get modtation return Mutation.mutate(model)
# maybe have a predefined inital network that a user can put in
# # TODO larning rate is a tensor so it can be adjusted during training (can pass in function) sess.run(train_step,learning_rate = tf.placeholder(tf.float32, shape=[]) feed_dict={learning_rate: 0.1})
#  TODO have all train related variables as functions to pass in, reg strength, learnin rate, dropout, these can be
# closures and can estimate internal params for functions based on feedback durning training

'''
stopping rule function! can be done for accuracy as well
def stopping_rule():
    loss = 100
    def c(new_loss):
        nonlocal loss
        old_loss, loss = loss, new_loss
        return loss < old_loss
    return c
'''

# How to get all subclasses for sc in Mutation.__subclasses__(): get_prob()
def run():
    tournament_report = {}  # or load previous
    error_logs = []
    for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
        print('Getting new Generation %d' % (generation))
        if CURRENT_GENERATION > 0:
            models_params = generate_mutated_models(tournament_report[generation - 1]['selected'])
        else:
            models_params = generate_initial_population()

        model_summaries = {}
        population = 0
        # TODO fix confusion between parameters and values for layers only values
        while population < POPULATION:
            # should be saved
            try:
                model, training_parameters = models_params.__next__()
                model.generation = CURRENT_GENERATION
                tf_model = TensorflowModel(model)
                trained_model, layer_parameters, summary = tf_model.run(DATASET, training_parameters)
                model_name = '%d_%d' % (generation, population)
                trained_model.name = model_name
                print('model, accuracy:', model_name, summary.validation_accuracy)
                print('model structure:', model_to_string(model))
                model_summaries[model_name] = summary
                model_path = os.path.join(DIR, generation, str(population) + '_model.p')
                params_path = os.path.join(DIR, generation, str(population) + '_params.p')
                with open(model_path, 'w') as model_file:
                    pickle.dump(trained_model, model_file)
                with open(params_path, 'w') as params_file:
                    pickle.dump(layer_parameters, params_file)
                population += 1
            except Exception as e:
                error_logs.append({'%d_%d' % (generation, population): str(e)})
        # TODO save report periodically so that it can be loaded if crash happens
        summary_path = os.path.join(DIR, generation, 'summary.p')
        with open(summary_path, 'w') as summary_file:
            pickle.dump(model_summaries, summary_file)
        tournament_report[generation]['summary'] = model_summaries

        selected = select_models(model_summaries)  # pass in selection function
        tournament_report[generation]['selected'] = selected

    report_path = os.path.join(DIR, 'report.p')
    with open(report_path, 'w') as report_file:
        pickle.dump(tournament_report, report_path)
    errors_path = os.path.join(DIR, 'errors.p')
    with open(report_path, 'w') as errors_file:
        pickle.dump(error_logs, errors_file)

    # TODO write test code to run test stats on final 5 models


def select_models(model_summaries):
    sorted_x_entropy = sorted(model_summaries.values(), key=lambda model_summary:
                                                                model_summary.validation_x_entropy)
    return {model_summary.name: model_summary for model_summary in sorted_x_entropy[:SELECT]}


def generate_initial_population():
    ''' Returns a generator that yeilds random models with
    one convolutional layer followed by a fully connected output layer

    '''
    layers_to_train = []
    filter_size = get_filter_size(IMAGE_SHAPE[0], IMAGE_SHAPE[1])
    filters = get_number_of_filters()

    layer = ConvolutionalLayer(filter_size=filter_size, filters=filters, name='c0')
    layers_to_train.append('c0')

    logits = OutputLayer()
    layers_to_train.append('logits')
    model = Model(convolutional_layers=[layer], logits=logits, image_shape=IMAGE_SHAPE, clasees=NUM_CLASSES)
    training_parameters = TrainingFunctions(layers_to_train=layers_to_train, iterations=interations_function,
                                            learning_rate=LEARNING_RATE)

    yield model, training_parameters


2


def memoize(obj):
    cache = obj.cache = {}
    functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return copy.deepcopy(cache[args])
    return memoizer

@memoize
def load_model(model_name):
    generation, number = model_name.split('_')
    model_path = os.path.join(DIR, generation, number + '_model.p')
    with open(model_path, 'r') as model_file:
        model = pickle.load(model_file)
    return model

@memoize
def load_saved_values(model_name):
    generation, number = model_name.split('_')
    params_path = os.path.join(DIR, generation, number + '_values.p')
    with open(params_path, 'r') as params_file:
        values = pickle.load(params_file)
    return values


def generate_mutated_models(summaries):
    seen = set()
    models_names = summaries.keys()
    random_model_name = np.random.choice(models_names)
    mutations = Mutation.__subclasses__()
    cross_overs = CrossOver.__subclasses__()
    mutation_probabilities = [mutation.get_probability(summaries[random_model_name]) for mutation in mutations]
    max_mutation_prob = max(mutation_probabilities)

    pairs  = itertools.combinations(models_names, 2)
    random_pair = np.random.choice(pairs)
    cross_over_probabilities = [cross_over.get_probability(summaries[random_pair[0]], summaries[random_pair[1]]) for
                                cross_over in cross_overs]
    max_co_prob = max(cross_over_probabilities)
    if max_co_prob > max_mutation_prob and random_pair not in seen:
        saved_model1, saved_values1 = load_model(random_pair[0]), load_saved_values(random_pair[0])
        saved_model2, saved_values2 = load_model(random_pair[1]), load_saved_values(random_pair[1])
        yield cross_overs[cross_over_probabilities.index(max_co_prob)].cross((saved_model1, saved_values1),(saved_model2, saved_values2))

    else:
        saved_model, saved_values = load_model(random_model_name), load_saved_values(random_model_name)
        yield mutations[mutations.index(max_mutation_prob)].mutate(saved_model, saved_values)


def load_summaries(generation):
    summary_path = os.path.join(DIR, generation, 'summary.p')
    with open(summary_path, 'r') as summary_file:
        summaries = pickle.load(summary_file)
    return summaries





def model_to_string(model):
    conv_layers_string = ' --> '.join('[%s, fs: %d, f: %d]' %
                                      (layer.name, layer.filter_size, layer.filters) for layer in
                                      model.convolutional_layers)
    dense_layers_string = ' --> '.join('[%s, h: %d]' %
                                       (layer.name, layer.hidden_units) for layer in
                                       model.dense_layers)
    return ' --> '.join([conv_layers_string, dense_layers_string, model.logits.name])

