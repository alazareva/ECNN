from namedlist import namedlist, FACTORY, NO_DEFAULT
from ecnn.defaults import  *

class SavedValues(dict):
	def __init__(self):
		self.name = None


OutputLayer = namedlist('OutputLayer', [('name', 'logits'), ('hidden_units', NUM_CLASSES),
										('training_history', [0]*MAX_GENERATIONS*2)], use_slots=True, default=None)

Model = namedlist('Model', [('generation', NO_DEFAULT),
							('convolutional_layers', FACTORY(list)),
							('dense_layers', FACTORY(list)),
							('logits', OutputLayer()),
							('name', NO_DEFAULT),
							('ancestor', NO_DEFAULT),
							('trainable_parameters', NO_DEFAULT),
							('learning_rate', NO_DEFAULT),
							('validation_accuracy', NO_DEFAULT),
							('mutation', NO_DEFAULT)], default=None)

ConvolutionalLayer = namedlist('ConvolutionalLayer', [('filter_size', NO_DEFAULT),
													  ('filters', NO_DEFAULT),
													  ('output_shape', NO_DEFAULT),
													  ('name', NO_DEFAULT),
													  ('max_pool', False),
													  ('training_history', [0]*MAX_GENERATIONS)], default=None)

DenseLayer = namedlist('DenseLayer', [('hidden_units', NO_DEFAULT),
									  ('name', NO_DEFAULT),
									  ('training_history', [0]*MAX_GENERATIONS)], default= None)


TrainingFunctions = namedlist('TrainingFunctions', [('batch_size', NO_DEFAULT),  # function
														('iterations', NO_DEFAULT ),  #function
														('learning_rate', LEARNING_RATE),
														('regularization', NO_DEFAULT),
														('stopping_rule', NO_DEFAULT),
														('keep_prob_conv', NO_DEFAULT),
														('keep_prob_dense', NO_DEFAULT)], default=None) #function

LayerValues = namedlist('LayerValues', ['weights', 'biases'], default=None)




