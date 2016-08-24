import tensorflow as tf
import numpy as np

from ga_nn.class_defs import *

class TensorflowModel(object):

	def __init__(self):
		# variables to optimize in current run
		# variable values to save after run complete

		# TODO separate method for test

	def run_model(self, dataset, model, training_parameters):

		convolutional_layers = model.convolutional_layers
		dense_layers = model.dense_layers
		image_shape = model.image_shape
		classes = model.classes
		training_set_size = training_parameters.training_set_size
		batch_size = training_parameters.batch_size

		layers_to_train = training_parameters.layers_to_train
		saved_parameters = training_parameters.saved_parameters
		compute_iterations = training_parameters.iterations

		model_summary = ModelSummary()

		variables_to_train = []
		variables_to_save = {}

		with tf.Graph().as_default():

			x = tf.placeholder(tf.float32, [None, np.prod(image_shape)], name='x')
			y_ =  tf.placeholder(tf.float32, [None, classes])
			height, width, channels = image_shape
			x = tf.reshape(x, shape=[-1, height, width, channels])

			current_tensor = x
			previous_output_shape = image_shape
			previous_output_size = channels

			# TODO refactor so that common code is shared
            # TODO needs to save output size for layer in case more conv layers appended
			for layer in convolutional_layers:

				stddev = np.sqrt(2.0 / np.prod(previous_output_shape))
				kernel_shape = [layer.filter_size, layer.filter_size, previous_output_size, layer.filters]

				model_summary.filters.append(layer.filter_size)
				model_summary.input_channels.append(previous_output_size)

				W, b = self.get_weights_biases(kernel_shape, layer, stddev)
				variables_to_save[layer.name+'_W'] = W
				variables_to_save[layer.name+'_b'] = b
				current_tensor = tf.nn.relu(tf.nn.bias_add(
					tf.nn.conv2d(current_tensor, W, strides=[1, 1, 1, 1], padding='SAME'), b))

				previous_output_shape = self.get_tensor_shape(current_tensor)
				previous_output_size = previous_output_shape[-1]
				layer.output_shape = previous_output_shape # record the output of this layer


				if layer.name in layers_to_train:
					variables_to_train.append(W)
					variables_to_train.append(b)


			current_tensor = tf.reshape(current_tensor, [-1, self.get_tensor_shape(current_tensor)])

			for layer in dense_layers:

				previous_output_shape = self.get_tensor_shape(current_tensor)
				previous_output_size = previous_output_shape[-1]

				stddev = np.sqrt(2.0 / np.prod(layer.previous_output_shape))
				shape = [previous_output_size, layer.hidden_units]

				W, b = self.get_weights_biases(shape, layer, stddev)
				variables_to_save[layer.name+'_W'] = W
				variables_to_save[layer.name+'_b'] = b
				current_tensor = tf.nn.relu(tf.nn.bias_add(tf.matmul(current_tensor, W), b)) #regularize

			if layer.name in layers_to_train:
					variables_to_train.append(W)
					variables_to_train.append(b)



			# logits
			layer = model.logits
			previous_output_shape = self.get_tensor_shape(current_tensor)
			previous_output_size = previous_output_shape[-1]


			stddev = np.sqrt(2.0 / np.prod(layer.previous_output_shape))
			shape = [previous_output_size, classes]
			W, b = self.get_weights_biases(shape, layer, stddev)
			variables_to_save[layer.name +'_W'] = W
			variables_to_save[layer.name + '_b'] = b

			if layer.name in layers_to_train:
				variables_to_train.append(W)
				variables_to_train.append(b)

			logits = tf.nn.relu(tf.nn.bias_add(tf.matmul(current_tensor, W), b))



			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))

			tf.add_to_collection('losses', cross_entropy)
			loss = tf.add_n(tf.get_collection('losses'))

			# can be modified to have multiple optimizers per layer (check stackoverflow)
			opt = tf.train.GradientDescentOptimizer(training_parameters.learning_rate)

			gradients = tf.gradients(loss, variables_to_train)
			train_op = opt.apply_gradients(zip(gradients, variables_to_train))
			# train_op = tf.group(train_op1, train_op2)

			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32))

			# Build an initialization operation to run below.
			init = tf.initialize_all_variables()

			# Start running operations on the Graph.


			with tf.Session() as sess:
				sess.run(init)

				number_of_params_to_train = self.compute_number_of_parameters(variables_to_train)

				max_epohs = compute_iterations(number_of_params_to_train)
				max_batches = int(training_set_size/batch_size)*max_epohs

				step = 0

				print "Training"
				while step < max_batches:
					try:
						batch_xs, batch_y = dataset.next_batch(batch_size)
						feed_dict = {y_:batch_y, x:batch_xs}
						_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

						self.check_nan(loss_value)
						step += 1
					except Exception as e:
						model.name = None
						return False, {'model': model, 'training_parameters': training_parameters, 'error': e}

				# Generate validation metric

				val_xs, val_y = dataset.X_val, dataset.y_val
				feed_dict = {y_:val_y, x: val_xs}
				validation_accuracy = sess.run(accuracy, feed_dict=feed_dict)

				# Save layers
				new_saved_parameters = {}

				# TODO trying to not use separate data structure
				for layer in convolutional_layers + dense_layers +[logits]:
					W = variables_to_save[layer.name + '_W']
					b = variables_to_save[layer.name + '_b']

					layer_parameters = ModelLayerParameters(W.eval(), b.eval())
					new_saved_parameters[layer.name] = layer_parameters

					if layer.name in layers_to_train:
						layer.training_history[model.generation] = max_epohs


				# Update model summary with information
				model_summary.validation_accuracy = validation_accuracy
				model_summary.number_of_trained_layers = len(layers_to_train)
				model_summary.number_of_trained_parameters = len(layers_to_train)
				model_summary.layer_counts = (len(convolutional_layers), len(dense_layers))
				model_summary.number_of_trained_parameters = number_of_params_to_train
				model_summary.number_of_parameters = self.compute_number_of_parameters(tf.trainable_variables())


			# TODO neesd to return training params that include info for all new layers, even not trained layers
			# TODO needs to return the dims of the conv layers
			return True, model, layer_parameters, model_summary



	def check_nan(self, loss_value):
		if np.isnan(loss_value):
			raise ValueError('Model diverged with loss = NaN')

	def get_tensor_shape(self, tensor):
		"""
		Args:
			tensor: a tensorflow Tensor object

		Returns: the tensor shape discarding the batch size dimention

		"""
		return tensor.get_shape().as_list()[1:]


	def compute_number_of_parameters(self, collection):
		return sum([np.prod(self.get_tensor_shape(tensor)) for tensor in collection])

	def get_weights_biases(self, shape, layer, stddev):




		if layer.weights: #if there's already weights 
			W = tf.Variable(layer.weights, name=layer.name + '_W', trainable=True)
			b = tf.Variable(layer.biases, name=layer.name + '_b', trainable=True)
		else:
			W = tf.Variable(tf.trucated_normal(shape, stddev=stddev), name=layer.name + '_W', trainable=True)
			b = tf.Variable(tf.zeros(shape[-1]), name = layer.name + '_b', trainable=True)

		return W, b


