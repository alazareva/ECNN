from namedlist import namedlist, FACTORY, NO_DEFAULT
import abc
import numpy as np
from ecnn.defaults import  *

OutputLayer = namedlist('OutputLayer', [('name', 'logits'), ('hidden_units', NUM_CLASSES),
										('training_history', NO_DEFAULT)], use_slots=True, default=None)

Model = namedlist('Model', [('generation', NO_DEFAULT),
							('convolutional_layers', FACTORY(list)),
							('dense_layers', FACTORY(list)),
							('logits', OutputLayer()),
							('name', NO_DEFAULT),
							('ancestor', NO_DEFAULT),
							('image_shape', IMAGE_SHAPE),
							('classes', NUM_CLASSES)], default=None)

ConvolutionalLayer = namedlist('ConvolutionalLayer', ['filter_size', 'filters', 'output_shape', 'name', 'training_history'], default=None)

DenseLayer = namedlist('DenseLayer', ['hidden_units', 'name', 'training_history'], default= None)



ModelSummary = namedlist('ModelSummary', [('name', NO_DEFAULT),
										  ('number_of_trained_layers', NO_DEFAULT),
										  ('validation_accuracy', NO_DEFAULT),
										  ('number_of_trained_parameters', NO_DEFAULT),
										  ('filters', FACTORY(list)), ('layer_counts', NO_DEFAULT),
										  ('trainable_parameters', NO_DEFAULT),
										  ('input_channels', FACTORY(list))], default= None)

TrainingInstructions = namedlist('TrainingInstructions', [('training_set_size', NO_DEFAULT),
														('batch_size', NO_DEFAULT),
														('layers_to_train', FACTORY(list)),
														('iterations', NO_DEFAULT ),
														('learning_rate', LEARNING_RATE),
														('saved_parameters', FACTORY(dict))], default=None)

# refactor saved params to be their own thing decouple from Training Params

LayerValues = namedlist('LayerValues', ['weights', 'biases'], default=None)

class SavedValues(dict):
    pass

# TODO have base classes for Mutation and Corss, each subclass has static method to compute probs and can
# be intantiated ith moded to return mutated models, main program has list of cls and can contruct them on the fly
# based on retuned probs, add @ cache wwhen loaded morels but make deep copies
# use abc to find all the subclasses


class Mutation(object):
	__metaclass__ = abc.ABCMeta

	@staticmethod
	@abc.abstractmethod
	def get_probability(model):
		"""Computes the probability of particular mutation using model information."""
		return

	@staticmethod
	@abc.abstractmethod
	def mutate(model):
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
	def update_values_and_instructions(model, saved_values, new_layer_index):
		layers_to_freeze =  np.random.randint(new_layer_index)
		new_saved_values = {}
		layers_to_train = []

		for i, layer in enumerate(model.convolutional_layers + model.dense_layers + [model.logits]):
			if i < new_layer_index:  # old layers use old parameters
				new_saved_values[layer.name] = saved_values[layer.name]
			if i > layers_to_freeze:
				layers_to_train.append(layer.name)

		training_instructions = TrainingInstructions(layers_to_train=layers_to_train)

		return training_instructions

	@staticmethod
	@abc.abstractmethod
	def cross(model1, model2):
		"""mutates the models returns the first one."""
		return

class AppendConvolutionalLayer(Mutation):
		@staticmethod
		def get_probability(model):
			"""Concrete the probability of particular mutation using model information."""
			return

		@staticmethod
		def mutate(model):
			"""mutates the model."""
			return

class AppendDenselLayer(Mutation):
		@staticmethod
		def get_probability(model):
			"""Concrete the probability of particular mutation using model information."""
			return
		@staticmethod
		def mutate(model):
			"""mutates the model."""
			return

class RemoveConvolutionalLayer(Mutation):
	@staticmethod
	def get_probability(model):
		"""Concrete the probability of particular mutation using model information."""
		return

	@staticmethod
	def mutate(model):
		"""mutates the model."""
		return



class RemoveDenselLayer(Mutation):
	@staticmethod
	def get_probability(model):
		"""Concrete the probability of particular mutation using model information."""
		return

	@staticmethod
	def mutate(model):
		"""mutates the model."""
		return

class Keep(Mutation):
	@staticmethod
	def get_probability(model):
		"""Concrete the probability of particular mutation using model information."""
		return

	@staticmethod
	def mutate(model):
		"""mutates the model."""
		return

class InsertConvolutionalLayer(Mutation):
	@staticmethod
	def get_probability(model):
		"""Concrete the probability of particular mutation using model information."""
		return

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
	def mutate(model):
		"""mutates the model."""
		return



class AdoptFilters(CrossOver):
	@staticmethod
	def get_probability(model1, model2):
		"""Computes the probability of particular cross over using model information."""
		if AdoptFilters.compatible_layers(model1, model2):
			return np.random.uniform(0.2,0.9)
		else:
			return 0
	@staticmethod
	def compatible_layers(model1, model2): #maybe model instead of model summary can be used and the other is run
		# summary
		filters1 = model1.filters
		filters2 = model2.filters
		channels1 = model1.input_channels
		channels2 = model2.input_channels
		x, y = np.where((np.absolute(filters1[:, np.newaxis] - filters2)
						 + np.absolute(channels1[:, np.newaxis] - channels2)) == 0)
		return list(zip(x, y))

	@staticmethod
	def cross(model_values1, model_values2):
		"""mutates the models returns the first one."""
		# model one gets model 2s weights but keeps it's other configurations
		model1, saved_values1 = model_values1
		model2, saved_values2 = model_values2

		candidate_layers = AdoptFilters.compatible_layers(model1, model2)
		layers = np.random.choice(candidate_layers)

		model1_layer_idx, model2_layer_idx = layers

		model1_layer = model1.convolutional_layers[model1_layer_idx]
		model2_layer = model2.convolutional_layers[model2_layer_idx]

		layer_values1 = saved_values1[model1_layer.name]
		layer_values2 = saved_values2.saved_parameters[model2_layer.name]

		new_weights = np.concatenate((layer_values1.weights, layer_values2.weights), axis=3)
		new_biases = np.concatenate((layer_values1.biases, layer_values2.biases), axis=0)

		model1.convolutional_layers[model1_layer_idx].filters = new_weights.shape[-1]
		saved_values1[model1_layer.name] = LayerValues(weights=new_weights, biases=new_biases)
		model1.ancestor = (model1.name, model2.name)
		new_layer_index = model1_layer_idx + 1

		values, instructions = super.update_values_and_instructions(model1, saved_values1, new_layer_index)

		return model1,

