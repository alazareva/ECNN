from unittest import TestCase
from ecnn.class_defs import  *

class TestAppend_dense_layer(TestCase):

    def test_append_dense_layer_to_conv(self):
        from ecnn.tournament import append_dense_layer
        c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        logits = OutputLayer()

        c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        mutation_params = {'hidden_units': 42}

        d0_expected = DenseLayer(hidden_units=42, name='d0')

        model = Model(convolutional_layers=[c0], logits=logits, name='test')

        actual = append_dense_layer(model, **mutation_params)
        expected = (Model(convolutional_layers=[c0_expected], dense_layers=[d0_expected], logits=logits, name='test'), 1)

        self.assertEqual(actual, expected)

    def test_append_dense_layer_to_dense(self):
        from ecnn.tournament import append_dense_layer
        c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        d0 = DenseLayer(hidden_units=42, name='d0')

        logits = OutputLayer()

        c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        d0_expected = DenseLayer(hidden_units=42, name='d0')



        mutation_params = {'hidden_units': 4}

        d1_expeted = DenseLayer(hidden_units=4, name='d1')


        model = Model(convolutional_layers=[c0], dense_layers=[d0], logits=logits, name='test')

        actual = append_dense_layer(model, **mutation_params)

        expected = (Model(convolutional_layers=[c0_expected], dense_layers=[d0_expected, d1_expeted], logits=logits,
                          name='test'), 2)

        self.assertEqual(actual, expected)