import itertools
import pickle
import os
import copy
import abc
import numpy as np
from collections import defaultdict
from scipy import stats

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
    """Returns a list of containing the number of filters for each convolutional layer in a model.

    Keyword arguments:
    model -- Model as defined in class_defs
    """
    return [l.filters for l  in model.convolutional_layers]

def input_channels(model):
    """Returns a list of containing the number of input channels for each convolutional layer in a model.

    Keyword arguments:
    model -- Model as defined in class_defs
    """
    ic = [IMAGE_SHAPE[-1]]
    for l in model.convolutional_layers[:-1]:
        ic.append(l.filters)
    return ic


def save(obj, filepath):
    """Saves an object using the specified filepath.

    Keyword arguments:
    obj -- a Python object
    filepath -- a filepath, must have a '.p' extention
    """
    with open(filepath, 'wb') as ofile:
        pickle.dump(obj, ofile)


def select_models(models, key_function, select):
    """Returns the best fit models.

    Keyword arguments:
    models -- a dictionary containing model names as keys and Models as values
    key_function -- the function used to sort the models in ascending order
    select -- the numbere of models to select
    """
    sorted_models = sorted(models.values(), key=key_function)
    return {model.name: model for model in sorted_models[-select:]}


def update_probabilities(models, key_function):
    """Updates the probability distribution parameters for mutations and layer sizes.

    Keyword arguments:
    models -- a dictionary containing model names as keys and Models as values
    key_function -- the function used to sort the models in ascending order
    """
    update_layer_value_probabilities(models, key_function)
    update_mutation_probabilities(models, key_function)

def update_layer_value_probabilities(models, key_function):
    """Updates the probability distribution parameters for layer sizes.

    Keyword arguments:
    models -- a dictionary containing model names as keys and Models as values
    key_function -- the function used to sort the models in ascending order
    """
    sorted_models = sorted(models.values(), key=key_function)
    top = sorted_models[-SELECT:]
    bottom = sorted_models[:-SELECT]
    update_layer_value_probs(top, bottom, ConvolutionalLayerUtils)
    update_layer_value_probs(top, bottom, DenseLayerUtils)


def restore_probabilities(mutation_alpha_beta, layer_alpha_beta):
    """Restores the probability parameters using serialized values.

    Keyword arguments:
    mutation_alpha_beta -- a dictionary with mutation names as keys and (alpha, beta) tuples as values
    layer_alpha_beta -- a dictionary with layer utilities names as keys and (alpha, beta) tuples as values
    """

    for m in Mutation.__subclasses__():
        m.alpha, m.beta = mutation_alpha_beta[m.__name__]
        print('restored', m.alpha, m.beta)
    for lu in LayerUtils.__subclasses__():
        lu.alpha, lu.beta = layer_alpha_beta[lu.__name__]
        print('restored', lu.alpha, lu.beta)

def update_layer_value_probs(bottom, top, cls):
    """Updates the probability distribution parameters for layer sizes.

    Keyword arguments:
    bottom -- the low performing models
    top -- the high performing models
    cls -- the layer utilities class corresponding to Concolutional Layers or Dense Layers
    """

    top = [cls.get_average_layer_size(model) for model in top if cls.has_layers(model)]
    bottom = [cls.get_average_layer_size(model) for model in bottom if cls.has_layers(model)]
    if top and bottom:
        _, p = stats.ttest_ind(top, bottom)
        if p < 0.05:
            top_mean = np.mean(top)
            bottom_mean = np.mean(bottom)
            print('adjusting parama', cls)
            if top_mean < bottom_mean:
                cls.beta += 2
            else:
                cls.alpha += 2
            print(cls.beta, cls.alpha)


def generate_initial_population():
    """Returns a generator tha yields models to make up the initial population
    The models consist of up to INITIAL_CONVOLUTIONAL_LAYERS number of convolutional layers.
    """
    while True:
        model = Model()
        number_of_initial_convo_layers = np.random.randint(2, INITIAL_CONVOLUTIONAL_LAYERS)
        input_shape = IMAGE_SHAPE.copy()
        for i in range(number_of_initial_convo_layers):
            filter_size, _ = ConvolutionalLayerUtils.get_filter_size(input_shape)
            filters = ConvolutionalLayerUtils.get_number_of_filters()
            pooling = ConvolutionalLayerUtils.is_max_pooling(input_shape)
            layer = ConvolutionalLayer(filter_size=filter_size, filters=filters, max_pool=pooling,  name='c%d' % i)
            model.convolutional_layers.append(layer)
            if pooling:
                input_shape[0] /= 2
                input_shape[1] /= 2
        logits = OutputLayer()
        model.logits = logits
        yield model, SavedValues()


def load_model(model_name):
    """Retuns a deep copy of a model loaded from a pickle file.

    Keyword arguments:
    model_name -- the name of the model
    """
    generation, number = model_name.split('_')
    filepath = os.path.join(DIR, str(generation), '%s_model.p' % number)
    model = load(filepath)
    return copy.deepcopy(model)


def copy_model(model):
    """Retuns a deep copy of a model removing unnessessary information.

    Keyword arguments:
    model -- a Model instance
    """
    m = copy.deepcopy(model)
    m.generation = None
    m.validation_accuracy = None
    m.ancestor = None
    m.trainable_parameters = None
    return m


def load_saved_values(model_name):
    """Retuns a deep copy of saved model parameters loaded from a pickle file.

    Keyword arguments:
    model_name -- the name of the model
    """
    generation, number = model_name.split('_')
    filepath = os.path.join(DIR, str(generation), '%s_values.p' % number)
    values = load(filepath)
    return copy.deepcopy(values)


def remove_values(all_models, selected):
    """Deletes pickle files containing model parameters for non-selected models.

    Keyword arguments:
    all_models -- a list containing the names of all models in a generation
    selected -- a list containing the names of all selected models in a generation
    """
    to_remove = set(all_models) - set(selected)
    for model_name in to_remove:
        generation, number = model_name.split('_')
        filepath = os.path.join(DIR, str(generation), '%s_values.p' % number)
        os.remove(filepath)

def generate_mutated_models(models, keep=True):
    """Returns a generator tha yields generated or crossed models to make up a new generation

    Keyword arguments:
    models -- a list containing the models to be mutated or crossed
    keep -- boolean indicating if the provided models should remain unchanged in the next generation
    """

    seen = set()
    models = list(models.values())
    if keep:
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
    """Returns an object loaded from the provided file path

    Keyword arguments:
    filepath -- the path to a picke file containing the object must have a '.p' extention
    """
    with open(filepath, 'rb') as ifile:
        obj = pickle.load(ifile)
    return obj


def update_mutation_probabilities(models, key_function):
    """Updates the alpha and beta parameters of the mutation probability distributions based on model
    performance

    Keyword arguments:
    models -- a dictionary with Models as values
    key_function -- the function used to rank models in terms of fit in ascending order
    """
    parents = {m.ancestor: m for m in models.values() if m.mutation == 'Keep'}

    rec = defaultdict(list)
    mutations = defaultdict(list)
    for model in models.values():
        if model.mutation != 'Keep' and model.mutation != 'AdoptFilters':
            rec[model.ancestor].append(model)

    for parent, children in rec.items():
        parent_eval =  key_function(parents[parent]) if parent in parents else 0
        for child in children:
            mutations[child.mutation].append(1 if key_function(child) >= parent_eval else 0)


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
    @staticmethod
    def get_remove_index(number_of_layers):
        """Returns the index of the layer to be removed

        Keyword arguments:
        number_of_layers -- the number of Convolutional or Dense layers in the network
        """
        assert number_of_layers > 0
        return number_of_layers - 1


class ConvolutionalLayerUtils(LayerUtils):
    alpha, beta = 1, 1

    @staticmethod
    def has_layers(model):
        """Returns a boolean indicating if a model is one or more convolutional layers

        Keyword arguments:
        model -- the Model
        """
        return model.convolutional_layers
    @staticmethod
    def get_average_layer_size(model):
        """Returns the average number of filters for all Convolutional Layers in the model

        Keyword arguments:
        model -- the Model
        """
        return np.mean([layer.filters for layer in model.convolutional_layers])
    @staticmethod
    def is_max_pooling(output_shape):
        """Returns a boolean indicating if pooling should be used, if the output shape is greater than 10 by 10
        pooling will be applied 30% of the time

        Keyword arguments:
        outout shape -- the shape of the output the convolution operation
        """
        height, width, _ = output_shape
        if height >= 10 and width >= 10:
            choice = np.random.randint(0, 100)
            if choice < 30:
                return True
        return False

    @staticmethod
    def get_number_of_filters():
        """Returns a randomly selected number of filters
        """
        return MIN_FILTERS+int(np.random.beta(ConvolutionalLayerUtils.alpha, ConvolutionalLayerUtils.beta)*(MAX_FILTERS-MIN_FILTERS))

    @staticmethod
    def get_filter_size(output_shape):
        """Returns the size of the square filters for a convolutional layer

        Keyword arguments:
        outout shape -- the shape of the output the previos layer
        """
        height, width, _ = output_shape
        min_height = min(max(int(height / 20), MIN_FILTER_SIZE), MAX_FILTER_SIZE)
        max_height = max(min(int(height / 4), MAX_FILTER_SIZE),MIN_FILTER_SIZE)
        height = np.random.randint(min_height, max_height)
        return (height, height)


class DenseLayerUtils(LayerUtils):
    alpha, beta = 1, 1

    @staticmethod
    def has_layers(model):
        """Returns a boolean indicating if a model is one or more dense layers

          Keyword arguments:
          model -- the Model
          """
        return model.dense_layers

    @staticmethod
    def get_average_layer_size(model):
        """Returns the average number of filters for all Dense Layers in the model

        Keyword arguments:
        model -- the Model
        """
        return np.mean([layer.hidden_units for layer in model.dense_layers])

    @staticmethod
    def get_desnse_layer_size():
        """Returns a randomly generated number of hidden units for a dense layer
        """
        return MIN_DENSE_LAYER_SIZE+int(np.random.beta(DenseLayerUtils.alpha,DenseLayerUtils.beta) *
                                                        (MAX_DENSE_LAYER_SIZE-MIN_DENSE_LAYER_SIZE))
class MutationUtils(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def update_values(model, saved_values, new_layer_index, removed=False):
        """Returns saved values for all layers not affected by an evolutionary operation

        Keyword arguments:
        model -- the Model
        saved_values -- the SavedValues for the model
        new_layer_index -- the index of the new layer or the first layer affect4ed by layer removal
        removed -- boolean indicating if a layer removal occured
        """

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
    alpha, beta = 1, 1

    @staticmethod
    @abc.abstractmethod
    def get_probability(model):
        """Computes the probability of particular mutation using model information."""
        return

    @staticmethod
    @abc.abstractmethod
    def mutate(model, saved_values):
        """Mutates the model."""
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
        """Crosses the two models"""
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
        filter_size, _ = ConvolutionalLayerUtils.get_filter_size(previous_layer.output_shape)
        filters = ConvolutionalLayerUtils.get_number_of_filters()
        pooling = ConvolutionalLayerUtils.is_max_pooling(previous_layer.output_shape)
        new_layer = ConvolutionalLayer(filter_size=filter_size,
                                       filters=filters, max_pool=pooling,
                                       name='c%d' % new_layer_index)
        model.convolutional_layers.append(new_layer)
        values= MutationUtils.update_values(model, saved_values, new_layer_index)
        model.mutation =   AppendConvolutionalLayer.__name__
        return model, values

class AppendDenseLayer(Mutation):
    @staticmethod
    def get_probability(model):
        if len(model.dense_layers) >= MAX_DENSE_LAYERS:
            return 0
        else:
            return np.random.beta(AppendDenseLayer.alpha, AppendDenseLayer.beta)

    @staticmethod
    def mutate(model, saved_values):
        hidden_units = DenseLayerUtils.get_desnse_layer_size() #evolving function
        new_layer_index = len(model.dense_layers)
        new_layer = DenseLayer(hidden_units = hidden_units,
                               name='d%d' % new_layer_index)
        model.dense_layers.append(new_layer)
        new_layer_index += len(model.convolutional_layers)
        values = MutationUtils.update_values(model, saved_values, new_layer_index)
        model.mutation =   AppendDenseLayer.__name__
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


class RemoveDenseLayer(Mutation):
    @staticmethod
    def get_probability(model):
        if len(model.dense_layers) == 0:
            return 0
        else:
            return np.random.beta(RemoveDenseLayer.alpha, RemoveDenseLayer.beta)

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
        model.mutation =   RemoveDenseLayer.__name__
        return model, values


# TODO don't need probabilities
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


class AdoptFilters(CrossOver):
    @staticmethod
    def get_probability(model1, model2):
        """Computes the probability of particular cross over using model information."""
        if AdoptFilters.compatible_layers(model1, model2):
            return np.random.uniform(0.2, 0.9)
        else:
            return 0
    @staticmethod
    def compatible_layers(model1, model2):
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

        new_weights = np.concatenate((layer_v1.weights, layer_v2.weights), axis=3)
        new_biases = np.concatenate((layer_v1.biases, layer_v2.biases), axis=0)

        m1.convolutional_layers[m1_layer_idx].filters = new_weights.shape[-1]
        v1[m1_layer.name] = LayerValues(weights=new_weights, biases=new_biases)

        new_layer_index = m1_layer_idx + 1

        values = MutationUtils.update_values(m1, v1, new_layer_index)
        m1.mutation =   AdoptFilters.__name__

        return m1, values
