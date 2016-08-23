from unittest import TestCase
from ga_nn.class_defs import  *

class TestRemove_convolutional_layer(TestCase):
    def test_remove_convolutional_layer_last(self):
        from ga_nn.tournament import remove_convolutional_layer

        c0 = ConvolutionalLayer(5, 11, [20, 20, 3], 'c0')
        c1 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c1')
        d0 = DenseLayer(6, 'd0')
        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 11, [20, 20, 3], 'c0')
        d0_expected = DenseLayer(6, 'd0')
        mutation_params = {'layer_index': 1}


        model = Model([c0, c1], [d0], logits, 'test', None, None, None)

        actual = remove_convolutional_layer(model, **mutation_params)
        expected = (Model([c0_expected], [d0_expected], logits, 'test', None, None, None), 1)

        self.assertEqual(actual, expected)


def test_remove_convolutional_layer_first(self):
    from ga_nn.tournament import remove_convolutional_layer

    c0 = ConvolutionalLayer(5, 11, [20, 20, 3], 'c0')
    c1 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c1')
    d0 = DenseLayer(6, 'd0')
    logits = Logits('logits')

    c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
    d0_expected = DenseLayer(6, 'd0')
    mutation_params = {'layer_index': 0}

    c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')

    model = Model([c0, c1], [d0], logits, 'test', None, None, None)

    actual = remove_convolutional_layer(model, **mutation_params)
    expected = (Model([c0_expected], [d0_expected], logits, 'test', None, None, None), 0)

    self.assertEqual(actual, expected)