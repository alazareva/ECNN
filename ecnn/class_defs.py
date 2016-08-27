from namedlist import namedlist, FACTORY, NO_DEFAULT
from ecnn.defaults import  *
from ecnn import mock_functions as functions
#from ecnn import functions

# TODO image shape and training set size are no
# TODO global config file


class SavedValues(dict):
	def __init__(self):
		self.name = None


OutputLayer = namedlist('OutputLayer', [('name', 'logits'), ('hidden_units', NUM_CLASSES),
										('training_history', [0]*MAX_GENERATIONS)], use_slots=True, default=None)

Model = namedlist('Model', [('generation', NO_DEFAULT),
							('convolutional_layers', FACTORY(list)),
							('dense_layers', FACTORY(list)),
							('logits', OutputLayer()),
							('name', NO_DEFAULT),
							('ancestor', NO_DEFAULT),
							('trainable_parameters', NO_DEFAULT),
							('validation_x_entropy', NO_DEFAULT),
							('validation_accuracy', NO_DEFAULT)], default=None)

ConvolutionalLayer = namedlist('ConvolutionalLayer', [('filter_size', NO_DEFAULT),
													  ('filters', NO_DEFAULT),
													  ('output_shape', NO_DEFAULT),
													  ('name', NO_DEFAULT),
													  ('training_history', [0]*MAX_GENERATIONS)], default=None)

DenseLayer = namedlist('DenseLayer', [('hidden_units', NO_DEFAULT),
									  ('name', NO_DEFAULT),
									  ('training_history', [0]*MAX_GENERATIONS)], default= None)


TrainingFunctions = namedlist('TrainingFunctions', [('batch_size', NO_DEFAULT),  # function
														('iterations', NO_DEFAULT ),  #function
														('learning_rate', LEARNING_RATE)], default=None) #function

LayerValues = namedlist('LayerValues', ['weights', 'biases'], default=None)




# TODO maybe refactor get probability into functions


