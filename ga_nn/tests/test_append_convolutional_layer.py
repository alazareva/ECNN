from unittest import TestCase
from ga_nn.class_defs import  *

class TestAppend_convolutional_layer(TestCase):

    def test_append_convolutional_layer_no_dense(self):
        from ga_nn.tournament import append_convolutional_layer

        c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        logits = OutputLayer('logits')

        c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        mutation_params = {'filter_size': 42, 'number_of_filters': 32}

        c1_expected = ConvolutionalLayer(filter_size=42, filters=32, name='c1')

        model = Model(convolutional_layers=[c0], logits=logits, name='test')

        actual = append_convolutional_layer(model, **mutation_params)
        expected = (Model(convolutional_layers=[c0_expected, c1_expected], logits=logits, name='test'), 1)

        self.assertEqual(actual, expected)


    def test_append_convolutional_layer_dense(self):
        from ga_nn.tournament import append_convolutional_layer

        c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        d0 = DenseLayer(hidden_units=6, name='d0')
        logits = OutputLayer()

        c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        d0_expected = DenseLayer(6,'d0')

        mutation_params = {'filter_size': 42, 'number_of_filters': 32}
        c1_expected = ConvolutionalLayer(filter_size=42, filters=32, name='c1')

        model = Model(convolutional_layers=[c0], dense_layers=[d0], logits=logits, name='test')

        actual = append_convolutional_layer(model, **mutation_params)
        expected = (Model(convolutional_layers=[c0_expected, c1_expected], dense_layers=[d0_expected], logits=logits,
                          name='test'), 1)

        self.assertEqual(actual, expected)

    def test_append_convolutional_layer_2(self):
        from ga_nn.tournament import append_convolutional_layer

        c0 = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        c1 = ConvolutionalLayer(filter_size=6, filters=4, output_shape=[2, 2, 30], name='c1')
        d0 = DenseLayer(hidden_units=6,name='d0')
        logits = OutputLayer()

        c0_expected = ConvolutionalLayer(filter_size=5, filters=10, output_shape=[20, 20, 3], name='c0')
        c1_expected = ConvolutionalLayer(filter_size=6, filters=4, output_shape=[2, 2, 30], name='c1')
        d0_expected = DenseLayer(6,'d0')

        mutation_params = {'filter_size': 10, 'number_of_filters': 20}
        c2_expected = ConvolutionalLayer(filter_size=10, filters=20, name='c2')

        model = Model(convolutional_layers=[c0, c1], dense_layers=[d0], logits=logits, name='test')

        actual = append_convolutional_layer(model, **mutation_params)
        expected = (Model(convolutional_layers=[c0_expected, c1_expected, c2_expected], dense_layers=[d0_expected],
                          logits=logits, name='test'), 2)

        self.assertEqual(actual, expected)





