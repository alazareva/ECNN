from unittest import TestCase
from ga_nn.class_defs import  *

class TestRemove_dense_layer(TestCase):


    def test_remove_dense_layer_last(self):
        from ga_nn.tournament import remove_dense_layer
        c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        d0 = DenseLayer(40, 'd0')
        d1 = DenseLayer(42, 'd1')

        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        d0_expected = DenseLayer(40, 'd0')

        mutation_params = {'layer_index': 1}

        model = Model([c0], [d0, d1], logits, 'test', None, None, None)

        actual = remove_dense_layer(model, **mutation_params)
        expected = (Model([c0_expected], [d0_expected], logits, 'test', None, None, None), 2)

        self.assertEqual(actual, expected)


def test_remove_dense_layer_last(self):
    from ga_nn.tournament import remove_dense_layer
    c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
    d0 = DenseLayer(40, 'd0')
    d1 = DenseLayer(42, 'd1')

    logits = Logits('logits')

    c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
    d0_expected = DenseLayer(42, 'd0')

    mutation_params = {'layer_index': 0}

    model = Model([c0], [d0, d1], logits, 'test', None, None, None)

    actual = remove_dense_layer(model, **mutation_params)
    expected = (Model([c0_expected], [d0_expected], logits, 'test', None, None, None), 2)

    self.assertEqual(actual, expected)
