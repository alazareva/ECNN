import itertools
import pickle
import os
import copy
import abc
import numpy as np
from collections import defaultdict

from ecnn.class_defs import *
from ecnn import functions

def model_to_string(model):
    conv_layers_string = ' --> '.join('[%s, fs: %d, f: %d, p: %r]' %
                                          (layer.name, layer.filter_size, layer.filters, layer.max_pool) for layer in
                                          model.convolutional_layers)

    dense_layers_string = ' --> '.join('[%s, h: %d]' %
                                       (layer.name, layer.hidden_units) for layer in
                                       model.dense_layers)
    return ' --> '.join([conv_layers_string, dense_layers_string, model.logits.name, str(model.ancestor),
                                                                                        str(model.mutation)])


def filters(model):
    return [l.filters for l  in model.convolutional_layers]

def input_channels(model):
    ic = [IMAGE_SHAPE[-1]]
    for l in model.convolutional_layers[:-1]:
        ic.append(l.filters)
    return ic


def values_for_testing(model):
    values = SavedValues() #need this to be SavedValues
    for layer in model.convolutional_layers+model.dense_layers + [model.logits]:
        values[layer.name] = []

    return values


def save(obj, filepath):
    with open(filepath, 'wb') as ofile:
        pickle.dump(obj, ofile)


def select_models(models):
    sorted_models = sorted(models.values(), key=functions.selection_function_val_accuracy)
    return {model.name: model for model in sorted_models[-SELECT:]}


def generate_initial_population():
    ''' Returns a generator that yeilds random models with
    one convolutional layer followed by a fully connected output layer

    '''
    while True:
        model = Model()
        number_of_initial_convo_layers = np.random.randint(2, INITIAL_CONVOLUTIONAL_LAYERS)
        input_shape = IMAGE_SHAPE.copy()
        for i in range(number_of_initial_convo_layers):
            filter_size, _ = LayerUtils.get_filter_size(input_shape)
            filters = LayerUtils.get_number_of_filters()
            pooling = LayerUtils.is_max_pooling(input_shape)
            layer = ConvolutionalLayer(filter_size=filter_size, filters=filters, max_pool=pooling,  name='c%d' % i)
            model.convolutional_layers.append(layer)
            if pooling:
                input_shape[0] /= 2
                input_shape[1] /= 2
        logits = OutputLayer()
        model.logits = logits
        yield model, SavedValues()


def load_model(model_name):
    generation, number = model_name.split('_')
    filepath = os.path.join(DIR, str(generation), '%s_model.p' % number)
    model = load(filepath)
    return copy.deepcopy(model)


def copy_model(model):
    m = copy.deepcopy(model)
    m.generation = None
    m.validation_accuracy = None
    m.validation_x_entropy = None
    m.ancestor = None
    m.trainable_parameters = None
    return m


def load_saved_values(model_name):
    generation, number = model_name.split('_')
    filepath = os.path.join(DIR, str(generation), '%s_values.p' % number)
    values = load(filepath)
    return copy.deepcopy(values)


def generate_mutated_models(models):
    seen = set()
    models = list(models.values())
    for model in models:
        yield Keep.keep(model)
    while True:
        m = models[np.random.choice(len(models))]
        mutations = Mutation.__subclasses__()
        cross_overs = CrossOver.__subclasses__()
        mutation_probabilities = [mutation.get_probability(m) for mutation in mutations]
        max_mutation_prob = max(mutation_probabilities)

        pairs = list(itertools.permutations(models, 2))
        m1, m2 = pairs[np.random.choice(len(pairs))]
        cross_over_probabilities = [cross_over.get_probability(m1, m2) for cross_over in cross_overs]
        max_co_prob = max(cross_over_probabilities)
        if max_co_prob > max_mutation_prob and (m1.name, m2.name) not in seen:
            v1, v2 = load_saved_values(m1.name), load_saved_values(m2.name)
            m1, m2 = copy_model(m1), copy_model(m2)
            seen.add((m1.name, m2.name))
            new_model, new_values =  cross_overs[cross_over_probabilities.index(max_co_prob)].cross((m1, v1), (m2, v2))
            new_model.ancestor = (m1.name, m2.name)
            yield new_model, new_values
        else:
            m, v = copy_model(m), load_saved_values(m.name)
            new_model, new_values = mutations[mutation_probabilities.index(max_mutation_prob)].mutate(m, v)
            new_model.ancestor = m.name
            yield new_model, new_values


def load_summaries(generation):
    summary_path = os.path.join(DIR, generation, 'summary.p')
    with open(summary_path, 'rb') as summary_file:
        summaries = pickle.load(summary_file)
    return summaries


def load(filepath):
    with open(filepath, 'rb') as ifile:
        obj = pickle.load(ifile)
    return obj


def update_mutation_probabilities(models):
    eval_func = functions.selection_function_val_accuracy
    parents = {m.ancestor: m for m in models.values() if m.mutation == 'Keep'} # just filter dictionary
    print('got parents', parents.keys())

    rec = defaultdict(list)
    mutations = defaultdict(list)
    for model in models.values():
        if model.mutation != 'Keep' and model.mutation != 'AdoptFilters':
            rec[model.ancestor].append(model)
    print(rec)

    for parent, children in rec.items():
        parent_eval =  eval_func(parents[parent]) if parent in parents else 0
        for child in children:
            mutations[child.mutation].append(1 if eval_func(child) >= parent_eval else 0)

    print(mutations)

    for mutation in Mutation.__subclasses__():
      if mutation.__name__ in mutations:
        mutation_results =  mutations[mutation.__name__]
        percent_success = np.mean(mutation_results)
        if percent_success >= 0.5:
            mutation.alpha *= 1+percent_success
            print('adjusted alpha for %s to %.3f' % (mutation.__name__, mutation.alpha))
        if percent_success <= 0.5:
            mutation.beta *= 1+(1-percent_success)
            print('adjusted beta for %s to %.3f' % (mutation.__name__, mutation.beta))


class LayerUtils(object):
    conv_alpa = 2
    conv_beta = 2
    dense_alpha = 2
    dense_beta = 2

    @staticmethod
    def get_filter_size(output_shape, square=True):
        height, width, _ = output_shape
        min_height = min(max(int(height / 20), MIN_FILTER_SIZE), MAX_FILTER_SIZE) #3
        max_height = max(min(int(height / 4), MAX_FILTER_SIZE),MIN_FILTER_SIZE) #8
        height = np.random.randint(min_height, max_height)
        return (height, height)

    @staticmethod
    def get_number_of_filters():  # for now returns squared filters but
        return np.random.randint(MIN_FILTERS, MAX_FILTERS)

    @staticmethod # TODO refactor this
    def is_max_pooling(output_shape):
        height, width, _ = output_shape
        if height >= 10 and width >= 10:
            choice = np.random.randint(0, 100)
            if choice < 30:
                return True
        return False

    @staticmethod
    def get_desnse_layer_size():
        return np.random.randint(MIN_DENSE_LAYER_SIZE, MAX_DENSE_LAYER_SIZE)

    @staticmethod
    def get_remove_index(number_of_layers):
        assert number_of_layers > 0
        return number_of_layers - 1

class MutationUtils(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def update_values(model, saved_values, new_layer_index, removed=False):
        new_saved_values = {}

        for i, layer in enumerate(model.convolutional_layers + model.dense_layers + [model.logits]):
            if i == new_layer_index:
                continue
            if i == new_layer_index + 1 and not removed:
                continue
            new_saved_values[layer.name] = saved_values[layer.name]

        return new_saved_values

class Mutation(object):
    __metaclass__ = abc.ABCMeta
    alpha = 2
    beta = 2
    @staticmethod
    @abc.abstractmethod
    def get_probability(model):
        """Computes the probability of particular mutation using model information."""
        return

    @staticmethod
    @abc.abstractmethod
    def mutate(model, saved_values):
        """mutates the model."""
        return


class CrossOver(object):
    __metaclass__ = abc.ABCMeta
    @staticmethod
    @abc.abstractmethod
    def get_probability(model1, model2):
        """Computes the probability of particular cross over using model information."""
        return

    @staticmethod
    @abc.abstractmethod
    def cross(model_values1, model_values2):
        return




class AppendConvolutionalLayer(Mutation):
    @staticmethod
    def get_probability(model):
        if len(model.convolutional_layers) >= MAX_CONVOLUTIONAL_LAYERS:
            return 0
        else:
            return np.random.beta(AppendConvolutionalLayer.alpha, AppendConvolutionalLayer.beta)

    @staticmethod
    def mutate(model, saved_values):
        new_layer_index = len(model.convolutional_layers)
        previous_layer = model.convolutional_layers[-1]
        filter_size, _ = LayerUtils.get_filter_size(previous_layer.output_shape)
        filters = LayerUtils.get_number_of_filters()
        pooling = LayerUtils.is_max_pooling(previous_layer.output_shape)
        new_layer = ConvolutionalLayer(filter_size=filter_size,
                                       filters=filters, max_pool=pooling,
                                       name='c%d' % new_layer_index)
        model.convolutional_layers.append(new_layer)
        values= MutationUtils.update_values(model, saved_values, new_layer_index)
        model.mutation =   AppendConvolutionalLayer.__name__
        return model, values

class AppendDenselLayer(Mutation):
    @staticmethod
    def get_probability(model):
        if len(model.dense_layers) >= MAX_DENSE_LAYERS:
            return 0
        else:
            return np.random.beta(AppendDenselLayer.alpha, AppendDenselLayer.beta)

    @staticmethod
    def mutate(model, saved_values):
        hidden_units = LayerUtils.get_desnse_layer_size() #evolving function
        new_layer_index = len(model.dense_layers)
        new_layer = DenseLayer(hidden_units = hidden_units,
                               name='d%d' % new_layer_index)
        model.dense_layers.append(new_layer)
        new_layer_index += len(model.convolutional_layers)
        values = MutationUtils.update_values(model, saved_values, new_layer_index)
        model.mutation =   AppendDenselLayer.__name__
        return model, values

class RemoveConvolutionalLayer(Mutation):
    @staticmethod
    def get_probability(model):
        if len(model.convolutional_layers) < 2:
            return 0
        else:
            return np.random.beta(RemoveConvolutionalLayer.alpha, RemoveConvolutionalLayer.beta)


    @staticmethod
    def mutate(model, saved_values):
        number_of_convolutional_layers = len(model.convolutional_layers)
        layer_index = LayerUtils.get_remove_index(number_of_convolutional_layers)
        model.convolutional_layers.pop(layer_index)
        # rename layers
        for i, layer in enumerate(model.convolutional_layers):
            layer.name = 'c%d' % i
        values = MutationUtils.update_values(model, saved_values, layer_index)
        model.mutation =   RemoveConvolutionalLayer.__name__
        return model, values


class RemoveDenselLayer(Mutation):
    @staticmethod
    def get_probability(model):
        if len(model.dense_layers) == 0:
            return 0
        else:
            return np.random.beta(RemoveDenselLayer.alpha, RemoveDenselLayer.beta)

    @staticmethod
    def mutate(model, saved_values):
        number_of_dense_layers = len(model.dense_layers)
        layer_index = LayerUtils.get_remove_index(number_of_dense_layers)
        model.dense_layers.pop(layer_index)
        # rename layers
        for i, layer in enumerate(model.dense_layers):
            layer.name = 'd%d' % i
        layer_index += len(model.convolutional_layers)
        values = MutationUtils.update_values(model, saved_values, layer_index)
        model.mutation =   RemoveDenselLayer.__name__
        return model, values


class Keep(object):
    @staticmethod
    def get_probability(model):
        if len(model.convolutional_layers) + len(model.dense_layers) + 1 \
                >= MAX_CONVOLUTIONAL_LAYERS + MAX_DENSE_LAYER_SIZE:
            return np.random.uniform(0.9, 0.95)
        else:
            return np.random.uniform(0.05, 0.8)
    @staticmethod
    def keep(model):
        m = copy_model(model)
        m.ancestor = m.name
        m.mutation = Keep.__name__
        return m, load_saved_values(m.name)


'''
class InsertConvolutionalLayer(Mutation):
	@staticmethod
	def get_probability(summary):
		conv_layers, _ = summary.layer_counts
		if conv_layers < MAX_CONVOLUTIONAL_LAYERS:
			return np.random.uniform(0.1, 0.2)
		else:
			pass

	@staticmethod
	def mutate(model):
		"""mutates the model."""
		return

class InsertDenseLayer(Mutation):
	@staticmethod
	def get_probability(model):
		"""Concrete the probability of particular mutation using model information."""
		return

	@staticmethod
	def mutate(model, saved_values):
		"""mutates the model."""
		return
'''


class AdoptFilters(CrossOver):
    @staticmethod
    def get_probability(model1, model2):
        """Computes the probability of particular cross over using model information."""
        if AdoptFilters.compatible_layers(model1, model2):
            return np.random.uniform(0.2,0.9) # for rarin
        else:
            return 0
    @staticmethod
    def compatible_layers(model1, model2): #maybe model instead of model summary can be used and the other is run
        # summary
        filters1 = np.array(filters(model1))
        filters2 = np.array(filters(model2))
        channels1 = np.array(input_channels(model1))
        channels2 = np.array(input_channels(model2))
        x, y = np.where((np.absolute(filters1[:, np.newaxis] - filters2)
                         + np.absolute(channels1[:, np.newaxis] - channels2)) == 0)
        return list(zip(x, y))

    @staticmethod
    def cross(model_values1, model_values2):
        """mutates the models returns the first one."""
        # model one gets model 2s weights but keeps it's other configurations
        m1, v1 = model_values1
        m2, v2 = model_values2

        candidate_layers = AdoptFilters.compatible_layers(m1, m2)
        layers = candidate_layers[np.random.choice(len(candidate_layers))]

        m1_layer_idx, m2_layer_idx = layers

        m1_layer = m1.convolutional_layers[m1_layer_idx]
        m2_layer = m2.convolutional_layers[m2_layer_idx]

        layer_v1 = v1[m1_layer.name]
        layer_v2 = v2[m2_layer.name]

        # for testing

        new_weights = np.concatenate((layer_v1.weights, layer_v2.weights), axis=3)
        new_biases = np.concatenate((layer_v1.biases, layer_v2.biases), axis=0)

        m1.convolutional_layers[m1_layer_idx].filters = new_weights.shape[-1]
        v1[m1_layer.name] = LayerValues(weights=new_weights, biases=new_biases)

        #m1.convolutional_layers[m1_layer_idx].filters = m1.convolutional_layers[
        # m1_layer_idx].filters+m2.convolutional_layers[m2_layer_idx].filters

        new_layer_index = m1_layer_idx + 1

        values = MutationUtils.update_values(m1, v1, new_layer_index)
        m1.mutation =   AdoptFilters.__name__

        return m1, values
