
# look at runtimes http://stackoverflow.com/questions/34293714/tensorflow-can-i-measure-the-execution-time-of-individual-operations
import re
import os.path
import tensorflow as tf
import numpy as np
import json




class TensorflowModel(object):

	def __init__(self):
		# variables to optimize in current run
		self.variables_to_train = []
		# variable values to save after run complete
		self.variables_to_save = {}

	def run_model(self, dataset, model, training_parameters):

		with tf.Graph().as_default():
			layers_to_train = training_parameters.layers_to_train

			x = tf.placeholder(tf.float32, [None, np.prod(training_parameters.image_shape)], name='x')
			y_ =  tf.placeholder(tf.float32, [None, training_parameters.num_classes])
			width, height, channels = training_parameters.image_shape
			x = tf.reshape(x, shape=[-1, height, width, channels])

			current_tensor = x
			previous_output_size = channels

			# TODO refactor so that common code is shared
            # TODO needs to save output size for layer in case more conv layers appended
			for layer in model.convolutional_layers:
				previous_output_shape = self.getTensorShape(current_tensor)
				previous_output_size = previous_output_shape[-1]
				stddev = np.sqrt(2.0 / np.prod(layer.previous_output_shape))
				kernel_shape = [layer.filter_size, layer.filter_size, previous_output_size, layer.filters]
				W, b = self.get_weights_biases(kernel_shape, layer, stddev)
				self.variables_to_save[layer.name+'_W'] = W
				self.variables_to_save[layer.name+'_b'] = b
				current_tensor = tf.nn.relu(tf.nn.bias_add(
					tf.nn.conv2d(current_tensor, W, strides=[1, 1, 1, 1], padding='SAME'), b))
				if layer.name in layers_to_train:
					self.variables_to_train.append(W)
					self.variables_to_train.append(b)
					layer.training_history[model.generation] = 1


			current_tensor = tf.reshape(current_tensor, [-1, self.getTensorShape(current_tensor)])

			for layer in model.dense_layers:
				previous_output_shape = self.getTensorShape(current_tensor)
				previous_output_size = previous_output_shape[-1]
				stddev = np.sqrt(2.0 / np.prod(layer.previous_output_shape))
				shape = [previous_output_size, layer.hidden_units]
				W, b = self.get_weights_biases(shape, layer, stddev)
				self.variables_to_save[layer.name+'_W'] = W
				self.variables_to_save[layer.name+'_b'] = b
				current_tensor = tf.nn.relu(tf.nn.bias_add(tf.matmul(current_tensor, W), b)) #regularize
			if layer.name in layers_to_train:
					self.variables_to_train.append(W)
					self.variables_to_train.append(b)
					layer.training_history[model.generation] = 1



			# logits
			layer = model.logits
			previous_output_shape = self.getTensorShape(current_tensor)
			previous_output_size = previous_output_shape[-1]
			stddev = np.sqrt(2.0 / np.prod(layer.previous_output_shape))
			shape = [previous_output_size, layer.classes]

			W, b = self.get_weights_biases(shape, layer, stddev)

			self.variables_to_save[layer.name +'_W'] = W
			self.variables_to_save[layer.name + '_b'] = b

			layer.training_history[model.generation] = 1 # multiply by itneations after
			if layer.name in layers_to_train:
				self.variables_to_train.append(W)
				self.variables_to_train.append(b)

			logits = tf.nn.relu(tf.nn.bias_add(tf.matmul(current_tensor, W), b))



			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))

			tf.add_to_collection('losses', cross_entropy)
			loss = tf.add_n(tf.get_collection('losses'))
			# can be modified to have multiple optimizers per layer (check stackoverflow)
			opt = tf.train.GradientDescentOptimizer(training_parameters.learning_rate)
			gradients = tf.gradients(loss, self.variables_to_train)
			train_op = opt.apply_gradients(zip(gradients, self.variables_to_train))
			# train_op = tf.group(train_op1, train_op2)

			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32))

			# Build an initialization operation to run below.
			init = tf.initialize_all_variables()

			# Start running operations on the Graph.

            # TODO needs context namager
			sess = tf.Session()

			sess.run(init)

            # TODO compute number of trainable parameters

			max_epohs = training_parameters.epohs
			training_set_size = training_parameters.training_set_size
			batch_size = training_parameters.batch_size

			max_batches = int(training_set_size/batch_size)*max_epohs

			step = 0

			print "Training"
			while step * batch_size  < max_batches:
				batch_xs, batch_y = dataset.next_batch(batch_size)
				feed_dict = {y_:batch_y, x:batch_xs}

				# TODO catch model crashes and file crash report
				_, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

				assert not np.isnan(loss_value), 'Model diverged with loss = NaN' #log this and quit
				step += 1

			# Generate validation metric

			val_xs, val_y = dataset.X_val, dataset.y_val
			feed_dict = {y_:val_y, x: val_xs}
			validation_accuracy = sess.run(accuracy, feed_dict=feed_dict)

			# Save layers
			for layer in model.convolutional_layers + model.dense_layers+[model.logits]:
				W = self.variables_to_save[layer.name + '_W']
				b = self.variables_to_save[layer.name + +'_b']
				layer.weights = W.eval()
				layer.biases = b.eval()
				layer.training_history[model.generation] *= max_epohs

			# Update model with information
			model.validation_accuracy = validation_accuracy
			model.number_of_parameters = None

			summary = {'accuracy': validation_accuracy,
					   'conv_filters': model.convolutional_layers[-1].filters,
					   'layer_counts': (len(model.convolutional_layers, len(model.dense_layers)))}

            # TODO neesd to return training params that include info for all new layers, even not trained layers
            # TODO needs to return the dims of the conv layers
			return True, model, saved_params, summary


	def getTensorShape(self, tensor):
		"""
		Args:
			tensor: a tensorflow Tensor object

		Returns: the tensor shape discarding the batch size dimention

		"""
		return tensor.get_shape().as_list()[1:]

	def get_weights_biases(self, shape, layer, stddev):




		if layer.weights: #if there's already weights 
			W = tf.Variable(layer.weights, name=layer.name + '_W')
			b = tf.Variable(layer.biases, name=layer.name + '_b')
		else:
			W = tf.Variable(tf.trucated_normal(shape, stddev=stddev), name=layer.name + '_W')
			b = tf.Variable(tf.zeros(shape[-1]), name = layer.name + '_b')

		return W, b


