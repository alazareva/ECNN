from unittest import TestCase
from ecnn.class_defs import  *

class TestRemove_dense_layer(TestCase):


    def test_remove_dense_layer_last(self):
        from ecnn.tournament import remove_dense_layer
        c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        d0 = DenseLayer(hidden_units=40, name='d0')
        d1 = DenseLayer(hidden_units=42, name='d1')

        logits = OutputLayer('logits')

        c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        d0_expected = DenseLayer(hidden_units=40, name='d0')

        mutation_params = {'layer_index': 1}

        model = Model(convolutional_layers=[c0], dense_layers=[d0, d1], logits=logits, name='test')

        actual = remove_dense_layer(model, **mutation_params)
        expected = (Model(convolutional_layers=[c0_expected], dense_layers=[d0_expected], logits=logits,
                          name='test'), 2)

        self.assertEqual(actual, expected)


def test_remove_dense_layer_last(self):
    from ecnn.tournament import remove_dense_layer
    c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
    d0 = DenseLayer(hidden_units=40, name='d0')
    d1 = DenseLayer(hidden_units=42, name='d1')

    logits = OutputLayer('logits')

    c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
    d0_expected = DenseLayer(hidden_units=42, name='d0')

    mutation_params = {'layer_index': 0}

    model = Model(convolutional_layers=[c0], dense_layers=[d0, d1], logits=logits, name='test')

    actual = remove_dense_layer(model, **mutation_params)
    expected = (Model(convolutional_layers=[c0_expected], dense_layers=[d0_expected], logits=logits, name='test'), 2)

    self.assertEqual(actual, expected)
