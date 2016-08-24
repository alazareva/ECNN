from unittest import TestCase
from ga_nn.class_defs import  *

class TestRemove_convolutional_layer(TestCase):
    def test_remove_convolutional_layer_last(self):
        from ga_nn.tournament import remove_convolutional_layer

        c0 = ConvolutionalLayer(filter_size=5, filters=11, output_shape=[20, 20, 3], name='c0')
        c1 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c1')
        d0 = DenseLayer(hidden_units=6, name='d0')
        logits = OutputLayer()

        c0_expected = ConvolutionalLayer(filter_size=5, filters=11, output_shape=[20, 20, 3], name='c0')
        d0_expected = DenseLayer(hidden_units=6, name='d0')
        mutation_params = {'layer_index': 1}


        model = Model(convolutional_layers=[c0, c1], dense_layers=[d0], logits=logits, name='test')

        actual = remove_convolutional_layer(model, **mutation_params)
        expected = (Model(convolutional_layers=[c0_expected], dense_layers=[d0_expected], logits=logits, name='test'), 1)

        self.assertEqual(actual, expected)


def test_remove_convolutional_layer_first(self):
    from ga_nn.tournament import remove_convolutional_layer

    c0 = ConvolutionalLayer(filter_size=5, filters=11, output_shape=[20, 20, 3], name='c0')
    c1 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c1')
    d0 = DenseLayer(6, 'd0')
    logits = OutputLayer('logits')

    c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
    d0_expected = DenseLayer(6, 'd0')
    mutation_params = {'layer_index': 0}

    c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')

    model = Model(convolutional_layers=[c0, c1], dense_layers=[d0], logits=logits, name='test')

    actual = remove_convolutional_layer(model, **mutation_params)
    expected = (Model(convolutional_layers=[c0_expected], dense_layers=[d0_expected], logits=logits, name='test'), 0)

    self.assertEqual(actual, expected)