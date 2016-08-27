import tensorflow as tf
import numpy as np

from ecnn.class_defs import *


class TensorflowModel(object):
    def __init__(self, model):

        self.model = model
        self.training_functions = None
        self.train_mode = False
        self.variables_to_save = {}
        self.model_summary = ModelSummary()
        self.saved_values = {}

    def run(self, dataset, saved_values, training_functions=None):

        convolutional_layers = self.model.convolutional_layers
        dense_layers = self.model.dense_layers
        image_shape = self.model.image_shape

        if  training_functions:
            self.training_functions = training_functions
            self.saved_values = saved_values
            self.train_mode = True

        with tf.Graph().as_default():

            x = tf.placeholder(tf.float32, [None, np.prod(image_shape)], name='x')
            y_ = tf.placeholder(tf.float32, [None, dataset.classes])
            height, width, channels = dataset.image_shape


            input_tensor = tf.reshape(x, shape=[-1, height, width, channels])

            print('input_shapen', input_tensor.get_shape().as_list())

            for layer in convolutional_layers:
                print(layer.name, input_tensor.get_shape().as_list())
                input_tensor = self.apply_convolution(layer, input_tensor)

            input_tensor = tf.reshape(input_tensor, [-1, np.prod(self.get_tensor_shape(input_tensor))])  # reshape before dense layers

            for layer in dense_layers:
                input_tensor = self.apply_dense_pass(layer, input_tensor)

            # logits
            layer = self.model.logits
            logits = self.apply_dense_pass(layer, input_tensor)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32))
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
            tf.add_to_collection('losses', cross_entropy)
            loss = tf.add_n(tf.get_collection('losses'))

            if self.train_mode:
                train_op = tf.train.GradientDescentOptimizer(self.training_functions.learning_rate).minimize(loss)
                init = tf.initialize_all_variables()

                with tf.Session() as sess:
                    sess.run(init)
                    batch_size = self.training_functions.batch_size()
                    number_of_params_to_train = self.compute_number_of_parameters(tf.trainable_variables())

                    max_epohs = self.training_functions.iterations(number_of_params_to_train)
                    max_batches = int(dataset.training_set_size / batch_size) * max_epohs

                    # TODO if batch size is dynamic, this will need to change

                    step = 0
                    print('Training')
                    while step < max_batches:
                        batch_xs, batch_y = dataset.next_batch(batch_size)
                        feed_dict = {y_: batch_y, x: batch_xs}
                        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                        print('%.5f' % loss_value)

                        self.check_nan(loss_value)
                        step += 1

                    # Generate validation metric
                    val_xs, val_y = dataset.X_val, dataset.y_val
                    feed_dict = {y_: val_y, x: val_xs}
                    validation_accuracy, validation_x_entropy = sess.run([accuracy, cross_entropy],
                                                                             feed_dict=feed_dict)
                    # Save layers
                    new_values = {}
                    for layer in convolutional_layers + dense_layers + [self.model.logits]:
                        W = self.variables_to_save[layer.name + '_W']
                        b = self.variables_to_save[layer.name + '_b']
                        layer_values = LayerValues(W.eval(), b.eval())
                        new_values[layer.name] = layer_values
                        if layer.name not in self.training_functions.layers_to_freeze:
                            layer.training_history[self.model.generation] = max_epohs

                    # Update model summary with information
                    self.model_summary.validation_accuracy = validation_accuracy
                    self.model_summary.validation_x_entropy = validation_x_entropy
                    self.model_summary.layer_counts = (len(convolutional_layers), len(dense_layers))
                    self.model_summary.trainable_parameters = number_of_params_to_train

                    return self.model, new_values, self.model_summary


            else:
                init = tf.initialize_all_variables()
                with tf.Session() as sess:
                    sess.run(init)
                    test_xs, test_y = dataset.X_test, dataset.y_test
                    feed_dict = {y_: test_y, x: test_xs}
                    test_accuracy = sess.run([accuracy], feed_dict=feed_dict)
                    return test_accuracy

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
        print('apply convolution to %s' % layer.name)

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
        stddev = np.sqrt(2.0 / np.prod(input_shape))
        shape = [input_size, layer.hidden_units]

        W, b = self.get_weights_biases(shape, layer, stddev)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)  # regularize

        if self.train_mode:
            self.add_variables_to_collections(layer.name, W, b)

        return output

    def add_variables_to_collections(self, name, W, b):
        self.variables_to_save[name + '_W'] = W
        self.variables_to_save[name + '_b'] = b

    def get_weights_biases(self, shape, layer, stddev):
        if layer.name in self.saved_values:  # if there's already weights
            print('restoring weights for %s' % layer.name)
            layer_parameters = self.saved_values[layer.name]
            W = tf.Variable(layer_parameters.weights, name=layer.name + '_W', trainable=True)
            b = tf.Variable(layer_parameters.biases, name=layer.name + '_b', trainable=True)
        elif self.train_mode:
            print('new weights for %s' % layer.name)
            W = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=layer.name + '_W', trainable=True)
            b = tf.Variable(tf.zeros(shape[-1]), name=layer.name + '_b', trainable=True)
        else:
            raise ValueError('No weights provided for %s layer %s' % (self.model, layer.name))
        return W, b


