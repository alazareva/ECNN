from namedlist import namedlist, FACTORY, NO_DEFAULT
from ecnn.defaults import  *

OutputLayer = namedlist('OutputLayer', [('name', 'logits'), ('hidden_units', NUM_CLASSES)], use_slots=True, default=None)

Model = namedlist('Model', [('generation', NO_DEFAULT),
							('convolutional_layers', FACTORY(list)),
							('dense_layers', FACTORY(list)),
							('logits', OutputLayer()),
							('name', NO_DEFAULT),
							('ancestor', NO_DEFAULT),
							('image_shape', IMAGE_SHAPE),
							('classes', NUM_CLASSES)], default=None)

ConvolutionalLayer = namedlist('ConvolutionalLayer', ['filter_size', 'filters', 'output_shape', 'name', 'training_history'], default=None)

DenseLayer = namedlist('DenseLayer', ['hidden_units', 'name'], default= None)



ModelSummary = namedlist('ModelSummary', [('name', NO_DEFAULT),
										  ('number_of_trained_layers', NO_DEFAULT),
										  ('validation_accuracy', NO_DEFAULT),
										  ('number_of_trained_parameters', NO_DEFAULT),
										  ('filters', FACTORY(list)), ('layer_counts', NO_DEFAULT),
										  ('trainable_parameters', NO_DEFAULT),
										  ('input_channels', FACTORY(list))], default= None)

TrainingParameters = namedlist('TrainingParameters', [('training_set_size', NO_DEFAULT),
													  ('batch_size', NO_DEFAULT),
													  ('layers_to_train', FACTORY(list)),
													  ('iterations', NO_DEFAULT ),
													  ('learning_rate', LEARNING_RATE),
													  ('saved_parameters', FACTORY(dict))], default=None)

LayerValues = namedlist('LayerValues', ['weights', 'biases'], default=None)

# TODO have base classes for Mutation and Corss, each subclass has static method to compute probs and can
# be intantiated ith moded to return mutated models, main program has list of cls and can contruct them on the fly
# based on retuned probs, add @ cache wwhen loaded morels but make deep copies