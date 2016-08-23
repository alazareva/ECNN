

# script to run tourtnament

import os
import pickle
import random
import itertools

import numpy as np


#from tensorflow_model import TensorflowModel

# nice way to filter dict dict(zip(selected_list, [summaries[model] for model in selected_list]))

# CRAZINESS how much randomness is allowed to happen, reusing convs from earlier layers

# ZOMBIE : bring back models that were mutated away from earier generations

from class_defs import  *

# TODO use coverage testing
# TODO refactor to take advantage of namedlist defaults
def run():
	tournament_report = {} # or load previous
	for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
		print 'Getting new Generation %d' %(generation)
		if CURRENT_GENERATION > 0:
			models_params = mutate(tournament_report[generation-1]['selected'])
		else:
			models_params = generate_initial_population()

		model_summaries = {}
		population = 0
		#TODO implement further iteration if crash occurs
		while population < POPULATION:
			model, training_parameters = models_params.next()
			tf_model = TensorflowModel()
			success, trained_model, layer_parameters, summary = tf_model.run(DATASET ,model, training_parameters)
			if success:
				model_name = '%d_%d' %(generation, population)
				trained_model.name = model_name
				print 'model, accuracy:', model_name, summary.validation_accuracy
				print 'model structure:', model_to_string(model)
				model_summaries[model_name] = summary
				model_path = os.path.join(DIR, generation, population + '_model.p')
				params_path = os.path.join(DIR, generation, population + '_params.p')
				with open(model_path, 'w') as model_file:
					pickle.dump(trained_model, model_file)
				with open(params_path, 'w') as params_file:
					pickle.dump(layer_parameters, params_file)
				population += 1
			else:
				print 'Error'
				#save errors

		# TODO save report periodically so that it can be loaded if crash happens
		summary_path = os.path.join(DIR, generation, 'summary.p')
		with open(summary_path, 'w') as summary_file:
			pickle.dump(model_summaries, summary_file)
		tournament_report[generation]['summary'] = model_summaries

		selected = select_models(model_summaries) # pass in selection function
		tournament_report[generation]['selected'] = selected

	report_path = os.path.join(DIR, 'report.p')
	with open(report_path, 'w') as report_file:
		pickle.dump(tournament_report, report_path)
	# TODO write test code to run test stats on final 5 models


def select_models(model_summaries):
	sorted_by_validation = sorted(model_summaries.vlues(), key=lambda model_summary: model_summary.validation_accuracy)
	return {model_summary.name:model_summary for model_summary in sorted_by_validation[:SELECT]}

def generate_initial_population():
	''' Returns a generator that yeilds random models with
	one convolutional layer followed by a fully connected output layer

	'''

	layers_to_train = []
	filter_size = get_filter_size(IMAGE_SHAPE[0], IMAGE_SHAPE[1])
	filters = get_number_of_filters()

	layer = ConvolutionalLayer(filter_size, filters, None, 'c0')
	layers_to_train.append('c0')

	logits = Logits('logits')
	layers_to_train.append('logits')
	model = Model([layer],[],logits, None, None, IMAGE_SHAPE, NUM_CLASSES)
	training_parameters = TrainingParameters(layers_to_train, interations_function, LEARNING_RATE,{})

	yield model, training_parameters


def generate_mutated_models(summaries):
	# takes a string list of model names
	# needs file paths
	# first figure out if crossovers are possible

	# keep track if cross over done
	seen = set()
	for mutation in _get_mutations(summaries):
		if mutation[0] == CROSS_MODELS and mutation not in seen:
			model1_name, model2_name = mutation[1]
			mutation_params = mutation[2]
			model_params1 = get_model_and_saved_parameters(model1_name)
			model_params2 = get_model_and_saved_parameters(model2_name)
			yield cross_models(model_params1, model_params2, **mutation_params)
		else:
			model_name = mutation[1]
			mutation_params = mutation[2]
			model, saved_parameters = get_model_and_saved_parameters(model_name)
			yield mutate(mutation[0], model, saved_parameters, **mutation_params)


def _get_mutations(summaries):
	possible_pairs = possible_cross_overs(summaries) # {(name1, name2):, [(layey1,layer2),...])}
	all_items =  [summary.name for summary in summaries] + possible_pairs.keys()
	random_model = random.choice(all_items)
	if len(random_model) == 2: # if it's a cross
		layers_to_cross = np.random.choice(possible_pairs[random_model])
		yield (CROSS_MODELS, random_model, {'layer_idxs':layers_to_cross})
	else:
		mutation, mutation_params = choose_mutation(summaries[random_model])
		yield (mutation, random_model, mutation_params)


def load_summaries(generation):
	summary_path = os.path.join(DIR, generation,'summary.p')
	with open(summary_path, 'r') as summary_file:
		summaries = pickle.load(summary_file)
	return summaries


def possible_cross_overs(summaries):
	# {(name1, name2):, [(layey1,layer2),...])}
	_possible_cross_overs = {}
	combinations = itertools.combinations(summaries.keys(), 2)
	for n1, n2 in combinations:
		filters1 = summaries[n1].filters
		filters2 = summaries[n2].filters
		channels1 = summaries[n1].input_channels
		channels2 = summaries[n2].input_channels
		x, y = np.where((np.absolute(filters1[:, np.newaxis] - filters2)
						 + np.absolute(channels1[:, np.newaxis] - channels2)) == 0)
		pairs = zip(x, y)
		if pairs:
			_possible_cross_overs[(n1, n2)] = pairs
			_possible_cross_overs[(n2, n1)] = zip(y, x)
	return _possible_cross_overs


# TODO implement the following
def choose_mutation(model_summary):
	# this show be a smarter function  that takes into accout model size
	c_layer_count, d_layer_count = model_summary.layer_counts
	# needs to return mutation, mutation_params



def get_model_and_saved_parameters(model_name):
	generation, number = model_name.split('_')
	model_path = os.path.join(DIR, generation, number + '_model.p')
	params_path = os.path.join(DIR, generation, number + '_params.p')
	with open(model_path, 'r') as model_file:
		model = pickle.load(model_file)
	with open(params_path, 'r') as params_file:
		params = pickle.load(params_file)
	return model, params


# TODO make this based on number of trainable parameters
def interations_function(number_of_training_parameters):
	return 100 # could change this to be based on params



def cross_models(model_params1, model_params2, **muation_params):
	# model one gets model 2s weights but keeps it's other configurations
	model1, params1 = model_params1
	model2, params2 = model_params2

	model1_layer_idx, model2_layer_idx = muation_params['layer_idxs']

	model1_layer = model1.convolutional_layers[model1_layer_idx]
	model2_layer = model2.convolutional_layers[model2_layer_idx]


	layer_parameters1 = params1.saved_parameters[model1_layer.name]
	layer_parameters2 = params2.saved_parameters[model2_layer.name]

	new_weights = np.concatenate((layer_parameters1.weights, layer_parameters2.weights), axis=3)
	new_biases = np.concatenate((layer_parameters1.biases, layer_parameters2.biases), axis=0)

	model1.convolutional_layers[model1_layer_idx].filters = new_weights.shape[-1]
	params1.saved_parameters[model1_layer.name] = ModelLayerParameters(new_weights, new_biases, np.zeros(MAX_GENERATIONS))
	new_layer_index = model1_layer_idx + 1
	model1.ancestor = (model1.name, model2.name)

	return update_training_parameters(model1, params1.saved_parameters, new_layer_index)


def mutate(mutation, model, saved_parameters, **muation_params):

	if mutation == KEEP:
		return model, saved_parameters
	if mutation == APPEND_CONVOLUTIONAL_LAYER:
		new_model, new_layer_index =  append_convolutional_layer(model,  **muation_params)
	if mutation == APPEND_DENSE_LAYER:
		new_model, new_layer_index = append_dense_layer(model,  **muation_params)
	if mutation == REMOVE_CONVOLUTIONAL_LAYER:
		new_model, new_layer_index = remove_convolutional_layer(model,  **muation_params)
	if mutation == REMOVE_DENSE_LAYER:
		new_model, new_layer_index = remove_dense_layer(model,  **muation_params)

	new_model.ancestor = model.name


	return update_training_parameters(new_model, saved_parameters)

def update_training_parameters(model, saved_parameters, new_layer_index):
	layers_to_freeze = get_layer_to_freeze(new_layer_index)
	new_saved_parameters = {}
	layers_to_train = []

	for i, layer in enumerate(model.convolutional_layers+ model.dense_layers+[model.logits]):
		if i < new_layer_index: # old layers use old parameters
			new_saved_parameters[layer.name] = saved_parameters[layer.name]
		if i > layers_to_freeze:
			layers_to_train.append(layer.name)

	new_training_parameters = TrainingParameters(layers_to_train, interations_function,
													 LEARNING_RATE, new_saved_parameters)

	return model, new_training_parameters



def remove_convolutional_layer(model, **muation_params):
	#number_of_convolutional_layers = len(model.convolutional_layers)
	#layer_index = get_remove_index(number_of_convolutional_layers)
	layer_index = muation_params['layer_index']
	model.convolutional_layers.pop(layer_index)
	#rename layers
	for i, layer in enumerate(model.convolutional_layers):
		layer.name = 'c%d' % i

	return model, layer_index


def remove_dense_layer(model, **muation_params):
	number_of_dense_layers = len(model.dense_layers)
	#layer_index = get_remove_index(number_of_dense_layers)
	layer_index = muation_params['layer_index']
	model.dense_layers.pop(layer_index)
	# rename layers
	for i, layer in enumerate(model.dense_layers):
			layer.name = 'd%d' % i
	layer_index += len(model.convolutional_layers)
	return model, layer_index


def get_remove_index(number_of_layers):
	assert number_of_layers > 0
	# need to add high prob for removing larger layers
	return number_of_layers - 1


def get_layer_to_freeze(new_layer_index):
	return np.random.randint(new_layer_index) #freeze can include everyting up to the new layer


def append_convolutional_layer(model , **muation_params):

	last_convolutional_layer = model.convolutional_layers[-1]
	height, width, depth = last_convolutional_layer.output_shape
	# filter_size = get_filter_size(height, width)
	filter_size = muation_params['filter_size']
	# number_of_filters = get_number_of_filters()
	number_of_filters = muation_params['number_of_filters']
	new_layer_index = len(model.convolutional_layers)
	new_layer = ConvolutionalLayer(filter_size, number_of_filters, None,'c%d'% new_layer_index)
	model.convolutional_layers.append(new_layer)

	return model, new_layer_index

def get_desnse_layer_size():
	return np.random.randint(MIN_DENSE_LAYER_SIZE, MAX_DENSE_LAYER_SIZE)

def append_dense_layer(model, **muation_params):
	#hidden_units = get_desnse_layer_size()
	hidden_units = muation_params['hidden_units']
	new_layer_index = len(model.dense_layers)
	new_layer = DenseLayer(hidden_units, 'd%d' % new_layer_index)
	model.dense_layers.append(new_layer)
	new_layer_index += len(model.convolutional_layers) #the layer index will be shifted

	return model, new_layer_index


def get_filter_size(height, width, square = True): # for now returns squared filters but
	min_height = max(int(height/20), MIN_FILTER_SIZE)
	max_height = min(int(height/2), MAX_FILTER_SIZE)
	return np.random.randint(min_height, max_height)


def get_number_of_filters():  # for now returns squared filters but
	return np.random.randint(MIN_FILTERS, MAX_FILTERS)



def model_to_string(model):
	conv_layers_string = ' --> '.join('[%s, fs: %d, f: %d]' %
									  (layer.name, layer.filter_size, layer.filters) for layer in
									  model.convolutional_layers)
	dense_layers_string = ' --> '.join('[%s, h: %d]' %
									  (layer.name, layer.hidden_units) for layer in
									  model.dense_layers)
	return ' --> '.join([conv_layers_string, dense_layers_string, model.logits.name])















