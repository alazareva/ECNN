from unittest import TestCase
from ga_nn.class_defs import  *

class TestAppend_dense_layer(TestCase):

    def test_append_dense_layer_to_conv(self):
        from ga_nn.tournament import append_dense_layer
        c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        mutation_params = {'hidden_units': 42}

        d0_expected = DenseLayer(42, 'd0')

        model = Model([c0], [], logits, 'test', None, None, None)

        actual = append_dense_layer(model, **mutation_params)
        expected = (Model([c0_expected], [d0_expected], logits, 'test', None, None, None), 1)

        self.assertEqual(actual, expected)

    def test_append_dense_layer_to_dense(self):
        from ga_nn.tournament import append_dense_layer
        c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        d0 = DenseLayer(42, 'd0')

        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        d0_expected = DenseLayer(42, 'd0')



        mutation_params = {'hidden_units': 4}

        d1_expeted = DenseLayer(4, 'd1')


        model = Model([c0], [d0], logits, 'test', None, None, None)

        actual = append_dense_layer(model, **mutation_params)

        expected = (Model([c0_expected], [d0_expected, d1_expeted], logits, 'test', None, None, None), 2)

        self.assertEqual(actual, expected)