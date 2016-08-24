from unittest import TestCase
import numpy as np
from ga_nn.class_defs import  *

class TestCross_models(TestCase):
    def test_cross_models_same_layer(self):
        from ga_nn.tournament import cross_models
        c0_1 = ConvolutionalLayer(filter_size=5, filters=3, output_shape=[20, 20, 3], name='c0')
        c0_2 = ConvolutionalLayer(filter_size=5, filters=2, output_shape=[20, 20, 3], name='c0')

        logits = Logits()

        model1 = Model(convolutional_layers=[c0_1], logits=logits, name='test1')
        model2 = Model(convolutional_layers=[c0_2], logits=logits, name='test2')

        w1 = np.ones((5, 5, 2, 3))
        w2 = np.ones((5, 5, 2, 2)) * 2

        b1 = np.zeros(3)
        b2 = np.zeros(2) * 2

        expected_w = np.concatenate((w1, w2), axis=3)
        expected_b = np.concatenate((b1, b2), axis=0)

        ml1 = ModelLayerParameters(weights=w1, biases=b1)
        ml2 = ModelLayerParameters(weights=w2, biases=b2)

        mp1 = TrainingParameters(saved_parameters={'c0': ml1})
        mp2 = TrainingParameters(saved_parameters={'c0': ml2})
        c0_expected = ConvolutionalLayer(filter_size=5, filters=5, output_shape=[20, 20, 3], name='c0')
        mutation_params = {'layer_idxs': (0,0)}

        actual_model, actual_mp = cross_models((model1, mp1),(model2, mp2),  **mutation_params)

        expected_model = Model(convolutional_layers=[c0_expected], logits=logits, name='test1', ancestor=('test1',
                                                                                                     'test2'))

        actual_ml = actual_mp.saved_parameters['c0']
        self.assertTrue((actual_ml.weights==expected_w).all())
        self.assertTrue((actual_ml.biases== expected_b).all())
        self.assertEquals((5, 5, 2, 5), actual_ml.weights.shape)
        self.assertEquals(actual_model, expected_model)


    def test_cross_models_different_layer(self):
        from ga_nn.tournament import cross_models
        c0_1 = ConvolutionalLayer(filter_size=5, filters=3, output_shape=[20, 20, 3], name='c0')
        c2_1 = ConvolutionalLayer(filter_size=6, filters=2, output_shape=[40, 40, 3], name='c1')

        c0_2 = ConvolutionalLayer(filter_size=6, filters=3, output_shape=[20, 20, 3], name='c0')

        logits = Logits('logits')

        model1 = Model(convolutional_layers=[c0_1, c2_1], logits=logits, name='test1')
        model2 = Model(convolutional_layers=[c0_2], logits=logits, name='test2')

        w1 = np.ones((6, 6, 2, 2))
        w2 = np.ones((6, 6, 2, 3)) * 2

        b1 = np.zeros(2)
        b2 = np.zeros(3) * 2

        expected_w = np.concatenate((w1, w2), axis=3)
        expected_b = np.concatenate((b1, b2), axis=0)

        ml1 = ModelLayerParameters(weights=w1, biases=b1)
        ml2 = ModelLayerParameters(weights=w2, biases=b2)

        mp1 = TrainingParameters(saved_parameters={'c0': None, 'c1': ml1})
        mp2 = TrainingParameters(saved_parameters={'c0': ml2})

        c0_expected = ConvolutionalLayer(filter_size=5, filters=3, output_shape=[20, 20, 3], name='c0')
        c1_expected = ConvolutionalLayer(filter_size=6, filters=5, output_shape=[40, 40, 3], name='c1')
        mutation_params = {'layer_idxs': (1,0)}

        actual_model, actual_mp = cross_models((model1, mp1),(model2, mp2),  **mutation_params)

        expected_model = Model(convolutional_layers=[c0_expected, c1_expected], logits=logits, name='test1',
                               ancestor=('test1', 'test2'))

        actual_ml = actual_mp.saved_parameters['c1']
        self.assertTrue((actual_ml.weights==expected_w).all())
        self.assertTrue((actual_ml.biases== expected_b).all())
        self.assertEquals((6, 6, 2, 5), actual_ml.weights.shape)
        self.assertEquals(actual_model, expected_model)