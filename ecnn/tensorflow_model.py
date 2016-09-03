import tensorflow as tf
import numpy as np


from ecnn.class_defs import *


class TensorflowModel(object):
    def __init__(self, model):

        self.model = model
        self.training_functions = None
        self.train_mode = False
        self.variables_to_save = {}
        self.saved_values = {}

    def run(self, dataset, saved_values, training_functions=None):

        convolutional_layers = self.model.convolutional_layers
        dense_layers = self.model.dense_layers
        image_shape = dataset.image_shape

        if  training_functions:
            self.training_functions = training_functions
            self.saved_values = saved_values
            self.train_mode = True

        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [None, np.prod(image_shape)], name='x')
            y_ = tf.placeholder(tf.float32, [None, dataset.classes])
            keep_prob_conv = tf.placeholder(tf.float32)
            keep_prob_dense = tf.placeholder(tf.float32)
            learning_rate = tf.placeholder(tf.float32)
            height, width, channels = dataset.image_shape


            input_tensor = tf.reshape(x, shape=[-1, height, width, channels])


            for layer in convolutional_layers:
                input_tensor = self.apply_convolution(layer, input_tensor, keep_prob=keep_prob_conv)

            input_tensor = tf.reshape(input_tensor, [-1, np.prod(self.get_tensor_shape(input_tensor))])  # reshape before dense layers

            for layer in dense_layers:
                input_tensor = self.apply_dense_pass(layer, input_tensor, keep_prob=keep_prob_dense) #,

            # logits
            layer = self.model.logits
            logits = self.get_logits(layer, input_tensor)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32))
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_))
            tf.add_to_collection('losses', cross_entropy)
            loss = tf.add_n(tf.get_collection('losses'))

            cross_entropy_average = tf.train.ExponentialMovingAverage(0.99)
            cross_entropy_average_op = cross_entropy_average.apply([cross_entropy])
            variable_averages = tf.train.ExponentialMovingAverage(0.9999) # TODO set the optional params
            variables_averages_op = variable_averages.apply(tf.trainable_variables())



            if self.train_mode:
                number_of_params_to_train = self.compute_number_of_parameters(tf.trainable_variables())
                with tf.control_dependencies([cross_entropy_average_op, variables_averages_op]):
                    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
                init = tf.initialize_all_variables()

                with tf.Session() as sess: # config=tf.ConfigProto(log_device_placement=True
                    sess.run(init)
                    batch_size = self.training_functions.batch_size()

                    max_epohs = self.training_functions.iterations(number_of_params_to_train)
                    batches_per_epoh = int(dataset.training_set_size / batch_size)
                    max_batches =  batches_per_epoh * max_epohs
                    stopping_rule = training_functions.stopping_rule()

                    # TODO if batch size is dynamic, this will need to change
                    k_p_c = training_functions.keep_prob_conv()
                    c_p_d = training_functions.keep_prob_dense()
                    lr_function = training_functions.learning_rate(0.5)
                    lr = lr_function(None)
                    step = 0
                    while step < max_batches:

                        batch_xs, batch_y = dataset.next_batch(batch_size)
                        feed_dict = {y_: batch_y, x: batch_xs, keep_prob_conv: k_p_c, keep_prob_dense: c_p_d,
                                     learning_rate: lr}

                        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                        self.check_nan(loss_value)
                        step += 1
                        if step % batches_per_epoh == 0: # TODO maybe revert old values if stopping rule
                            cross_entropy_value = sess.run(cross_entropy, feed_dict=feed_dict)
                            lr = lr_function(cross_entropy_value)
                            val_xs, val_y = dataset.X_val, dataset.y_val
                            feed_dict = {y_: val_y, x: val_xs, keep_prob_conv: 1.0, keep_prob_dense: 1.0}
                            val_acc = sess.run(accuracy, feed_dict=feed_dict)
                            if stopping_rule(val_acc):
                                 print('stopping rule')
                                 break


                    # Generate validation metric
                    val_xs, val_y = dataset.X_val, dataset.y_val
                    feed_dict = {y_: val_y, x: val_xs, keep_prob_conv: 1.0,keep_prob_dense: 1.0}
                    validation_accuracy, validation_x_entropy = sess.run([accuracy, cross_entropy],
                                                                             feed_dict=feed_dict)
                    # Save layers
                    new_values = SavedValues()
                    for layer in convolutional_layers + dense_layers + [self.model.logits]:
                        W = variable_averages.average(self.variables_to_save[layer.name + '_W'])
                        b = variable_averages.average(self.variables_to_save[layer.name + '_b'])
                        layer_values = LayerValues(W.eval(), b.eval())
                        new_values[layer.name] = layer_values
                        layer.training_history[self.model.generation] = max_epohs

                    # Update model summary with information
                    self.model.validation_accuracy = validation_accuracy
                    self.model.validation_x_entropy = cross_entropy_average.average(cross_entropy).eval()
                    self.model.trainable_parameters = number_of_params_to_train

                    return self.model, new_values


            else:
                init = tf.initialize_all_variables()
                with tf.Session() as sess:
                    sess.run(init)
                    test_xs, test_y = dataset.X_test, dataset.y_test
                    feed_dict = {y_: test_y, x: test_xs, keep_prob_conv: 1.0,
                                 keep_prob_dense: 1.0}
                    test_accuracy = sess.run([accuracy], feed_dict=feed_dict)
                    return test_accuracy

    def check_nan(self, loss_value):
        if np.isnan(loss_value) or np.isinf(loss_value):
            raise ValueError('Model diverged with loss = NaN or inf')

    def get_tensor_shape(self, tensor):
        """
        Args:
            tensor: a tensorflow Tensor object

        Returns: the tensor shape discarding the batch size dimention

        """
        return tensor.get_shape().as_list()[1:]

    def compute_number_of_parameters(self, collection):
        return sum([np.prod(self.get_tensor_shape(tensor)) for tensor in collection])

    def apply_convolution(self, layer, input_tensor, keep_prob = None):

        input_shape = self.get_tensor_shape(input_tensor)
        input_size = input_shape[-1]

        stddev = np.sqrt(2.0 / np.prod(input_shape))
        kernel_shape = [layer.filter_size, layer.filter_size, input_size, layer.filters]
        W, b = self.get_weights_biases(kernel_shape, layer, stddev)
        output = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(input_tensor, W, strides=[1, 1, 1, 1], padding='SAME'), b))

        if layer.max_pool:
            output = self.conv2d(output)

        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob)

        output_shape = self.get_tensor_shape(output)  # maybe do this after cuz it's in saved

        if self.train_mode:
            layer.output_shape = output_shape
            self.add_variables_to_collections(layer.name, W, b)

        return output


    def conv2d(self, input_tensor):
        return tf.nn.max_pool(input_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME')


    def apply_dense_pass(self, layer, input_tensor, keep_prob=None):
        logits = self.get_logits(layer, input_tensor)
        output = tf.nn.relu(logits)  # regularize
        if keep_prob is not None:
            output = tf.nn.dropout(output, keep_prob)
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
        if self.train_mode:
            weight_decay = tf.mul(tf.nn.l2_loss(W), self.training_functions.regularization(), name='weight_loss')
            tf.add_to_collection('losses', weight_decay)

    def get_weights_biases(self, shape, layer, stddev):
        if layer.name in self.saved_values:  # if there's already weights
            layer_values = self.saved_values[layer.name]
            W = tf.Variable(layer_values.weights, name=layer.name + '_W', trainable=True)
            b = tf.Variable(layer_values.biases, name=layer.name + '_b', trainable=True)
        elif self.train_mode:
            W = tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=layer.name + '_W', trainable=True)
            b = tf.Variable(tf.zeros(shape[-1]), name=layer.name + '_b', trainable=True)
        else:
            raise ValueError('No weights provided for %s layer %s' % (self.model, layer.name))
        return W, b


