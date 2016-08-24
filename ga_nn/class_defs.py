from namedlist import namedlist, FACTORY, NO_DEFAULT
from ga_nn.defaults import  *

Logits = namedlist('Logits', [('name', 'logits'), ('hidden_units', NUM_CLASSES)],  se_slots=True, default=None)

Model = namedlist('Model', [('generation', NO_DEFAULT),
							('convolutional_layers', FACTORY(list)),
							('dense_layers', FACTORY(list)),
							('logits', Logits()),
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

ModelLayerParameters = namedlist('ModelLayerParameters', ['weights', 'biases'], default=None)