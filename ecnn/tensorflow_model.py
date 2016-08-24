import tensorflow as tf
import numpy as np

from ecnn.class_defs import *


class TensorflowModel(object):
    def __init__(self, model, training_parameters):

        self.model = model
        self.training_parameters = None
        self.train_mode = False
        self.variables_to_save = {}
        self.variables_to_train = []
        self.model_summary = ModelSummary()

    def run_model(self, dataset, train_test_input):

        convolutional_layers = self.model.convolutional_layers
        dense_layers = self.model.dense_layers
        image_shape = self.model.image_shape

        if self.is_train_mode(train_test_input):
            self.training_parameters = train_test_input
            self.saved_parameters = self.training_parameters.saved_parameters
            self.train_mode = True

        with tf.Graph().as_default():

            x = tf.placeholder(tf.float32, [None, np.prod(image_shape)], name='x')
            y_ = tf.placeholder(tf.float32, [None, self.model.classes])
            height, width, channels = self.model.image_shape
            x = tf.reshape(x, shape=[-1, height, width, channels])

            input = x

            for layer in convolutional_layers:
                input = self.apply_convolution(layer, input)

                input = tf.reshape(input, [-1, np.prod(self.get_tensor_shape(input))])  # reshape before dense layers

            for layer in dense_layers:
                input = self.apply_dense_pass(layer, input)

            # logits
            layer = self.model.logits
            logits = self.apply_convolution(layer, input)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32))

            if self.train_mode:
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
                tf.add_to_collection('losses', cross_entropy)
                loss = tf.add_n(tf.get_collection('losses'))
                # can be modified to have multiple optimizers per layer (check stackoverflow)
                opt = tf.train.GradientDescentOptimizer(self.training_parameters.learning_rate)

                gradients = tf.gradients(loss, self.variables_to_train)
                train_op = opt.apply_gradients(zip(gradients, self.variables_to_train))
                # train_op = tf.group(train_op1, train_op2)


                init = tf.initialize_all_variables()

                with tf.Session() as sess:
                    sess.run(init)
                    batch_size = self.training_parameters.batch_sise
                    number_of_params_to_train = self.compute_number_of_parameters(self.variables_to_train)

                    max_epohs = self.training_parameters.iterations(number_of_params_to_train)
                    max_batches = int(self.training_parameters.training_set_size / batch_size) * max_epohs

                    step = 0

                    print('Training')
                    while step < max_batches:
                        batch_xs, batch_y = dataset.next_batch(batch_size)
                        feed_dict = {y_: batch_y, x: batch_xs}
                        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

                        self.check_nan(loss_value)
                        step += 1

                    # Generate validation metric
                    val_xs, val_y = dataset.X_val, dataset.y_val
                    feed_dict = {y_: val_y, x: val_xs}
                    validation_accuracy = sess.run(accuracy, feed_dict=feed_dict)

                    # Save layers
                    new_values = {}
                    for layer in convolutional_layers + dense_layers + [self.model.logits]:
                        W = self.variables_to_save[layer.name + '_W']
                        b = self.variables_to_save[layer.name + '_b']
                        layer_values = LayerValues(W.eval(), b.eval())
                        new_values[layer.name] = layer_values
                        if layer.name in self.layers_to_train:
                            layer.training_history[self.model.generation] = max_epohs

                    # Update model summary with information
                    self.model_summary.validation_accuracy = validation_accuracy
                    self.model_summary.number_of_trained_layers = len(self.layers_to_train)
                    self.model_summary.layer_counts = (len(convolutional_layers), len(dense_layers))
                    self.model_summary.number_of_trained_parameters = number_of_params_to_train
                    self.model_summary.number_of_parameters = self.compute_number_of_parameters(
                        tf.trainable_variables())

                    return True, self.model, new_values, self.model_summary


            else: # TODO need to pass in saved params for testing
                init = tf.initialize_all_variables()
                with tf.Session() as sess:
                    sess.run(init)
                    test_xs, test_y = dataset.X_test, dataset.y_test
                    feed_dict = {y_: test_y, x: test_xs}
                    test_accuracy, loss = sess.run([accuracy, loss], feed_dict=feed_dict)
                    return test_accuracy, loss

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

    def apply_convolution(self, layer, input_tensor):

        input_shape = self.get_tensor_shape(input_tensor)
        input_size = input_shape[-1]

        stddev = np.sqrt(2.0 / np.prod(input_shape))
        kernel_shape = [layer.filter_size, layer.filter_size, input_size, layer.filters]
        W, b = self.get_weights_biases(kernel_shape, layer, stddev)
        output = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding='SAME'), b))

        output_shape = self.get_tensor_shape(output)  # maybe do this after cuz it's in saved

        if self.train_mode:
            self.model_summary.filters.append(layer.filter_size)
            self.model_summary.input_channels.append(input_size)
            layer.output_shape = output_shape
            self.add_variables_to_collections(layer.name, W, b)

        return output

    # TODO add reguralization
    def apply_dense_pass(self, layer, input_tensor):
        logits = self.get_logits(layer, input_tensor)
        output = tf.nn.relu(logits)  # regularize
        return output

    def get_logits(self, layer, input_tensor):
        input_shape = self.get_tensor_shape(input_tensor)
        input_size = input_shape[-1]
        stddev = np.sqrt(2.0 / np.prod(layer.previous_output_shape))
        shape = [input_size, layer.hidden_units]

        W, b = self.get_weights_biases(shape, layer, stddev)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)  # regularize

        if self.train_mode:
            self.add_variables_to_collections(layer.name, W, b)

        return output

    def add_variables_to_collections(self, name, W, b):
        self.variables_to_save[name + '_W'] = W
        self.variables_to_save[name + '_b'] = b
        if name in self.training_parameters.layers_to_train:
            self.variables_to_train.append(W)
            self.variables_to_train.append(b)

    def get_weights_biases(self, shape, layer, stddev):

        if layer.name in self.saved_parameters:  # if there's already weights
            layer_parameters = self.saved_parameters[layer.name]
            W = tf.Variable(layer_parameters.weights, name=layer.name + '_W', trainable=True)
            b = tf.Variable(layer_parameters.biases, name=layer.name + '_b', trainable=True)
        elif self.train_mode:
            W = tf.Variable(tf.trucated_normal(shape, stddev=stddev), name=layer.name + '_W', trainable=True)
            b = tf.Variable(tf.zeros(shape[-1]), name=layer.name + '_b', trainable=True)
        else:
            raise ValueError('No weights provided for %s layer %s' % (self.model, layer.name))
        return W, b

    def is_train_mode(self, train_test_input):
        return train_test_input.__class__.__name__ == 'TrainingParameters'
