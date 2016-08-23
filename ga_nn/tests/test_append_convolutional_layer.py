from unittest import TestCase
from ga_nn.class_defs import  *

class TestAppend_convolutional_layer(TestCase):

    def test_append_convolutional_layer_no_dense(self):
        from ga_nn.tournament import append_convolutional_layer

        c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        mutation_params = {'filter_size': 42, 'number_of_filters': 32}

        c1_expected = ConvolutionalLayer(42, 32, None, 'c1')

        model = Model([c0], [], logits, 'test', None, None, None)

        actual = append_convolutional_layer(model, **mutation_params)
        expected = (Model([c0_expected, c1_expected], [], logits, 'test', None, None, None), 1)

        self.assertEqual(actual, expected)


    def test_append_convolutional_layer_dense(self):
        from ga_nn.tournament import append_convolutional_layer

        c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        d0 = DenseLayer(6,'d0')
        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        d0_expected = DenseLayer(6,'d0')

        mutation_params = {'filter_size': 42, 'number_of_filters': 32}
        c1_expected = ConvolutionalLayer(42, 32, None, 'c1')

        model = Model([c0], [d0], logits, 'test', None, None, None)

        actual = append_convolutional_layer(model, **mutation_params)
        expected = (Model([c0_expected, c1_expected], [d0_expected], logits, 'test', None, None, None), 1)

        self.assertEqual(actual, expected)

    def test_append_convolutional_layer_2(self):
        from ga_nn.tournament import append_convolutional_layer

        c0 = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        c1 = ConvolutionalLayer(6, 4, [2, 2, 30], 'c1')
        d0 = DenseLayer(6,'d0')
        logits = Logits('logits')

        c0_expected = ConvolutionalLayer(5, 10, [20, 20, 3], 'c0')
        c1_expected = ConvolutionalLayer(6, 4, [2, 2, 30], 'c1')
        d0_expected = DenseLayer(6,'d0')

        mutation_params = {'filter_size': 10, 'number_of_filters': 20}
        c2_expected = ConvolutionalLayer(10, 20, None, 'c2')

        model = Model([c0, c1], [d0], logits, 'test', None, None, None)

        actual = append_convolutional_layer(model, **mutation_params)
        expected = (Model([c0_expected, c1_expected, c2_expected], [d0_expected], logits, 'test', None, None, None), 2)

        self.assertEqual(actual, expected)





