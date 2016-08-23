from unittest import TestCase
import numpy as np
from ga_nn.class_defs import  *

class TestCross_models(TestCase):
    def test_cross_models_same_layer(self):
        from ga_nn.tournament import cross_models
        c0_1 = ConvolutionalLayer(5, 3, [20, 20, 3], 'c0')
        c0_2 = ConvolutionalLayer(5, 2, [20, 20, 3], 'c0')

        logits = Logits('logits')

        model1 = Model([c0_1], [], logits, 'test1', None, None, None)
        model2 = Model([c0_2], [], logits, 'test2', None, None, None)

        w1 = np.ones((5, 5, 2, 3))
        w2 = np.ones((5, 5, 2, 2)) * 2

        b1 = np.zeros(3)
        b2 = np.zeros(2) * 2

        expected_w = np.concatenate((w1, w2), axis=3)
        expected_b = np.concatenate((b1, b2), axis=0)

        ml1 = ModelLayerParameters(w1, b1, None)
        ml2 = ModelLayerParameters(w2, b2, None)

        mp1 = TrainingParameters([], None, None, {'c0': ml1})
        mp2 = TrainingParameters([], None, None, {'c0': ml2})
        c0_expected = ConvolutionalLayer(5, 5, [20, 20, 3], 'c0')
        mutation_params = {'layer_idxs': (0,0)}

        actual_model, actual_mp = cross_models((model1, mp1),(model2, mp2),  **mutation_params)

        expected_model = Model([c0_expected], [], logits, 'test1', ('test1', 'test2'), None, None)

        actual_ml = actual_mp.saved_parameters['c0']
        self.assertTrue((actual_ml.weights==expected_w).all())
        self.assertTrue((actual_ml.biases== expected_b).all())
        self.assertEquals((5, 5, 2, 5), actual_ml.weights.shape)
        self.assertEquals(actual_model, expected_model)


    def test_cross_models_different_layer(self):
        from ga_nn.tournament import cross_models
        c0_1 = ConvolutionalLayer(5, 3, [20, 20, 3], 'c0')
        c2_1 = ConvolutionalLayer(6, 2, [40, 40, 3], 'c1')

        c0_2 = ConvolutionalLayer(6, 3, [20, 20, 3], 'c0')

        logits = Logits('logits')

        model1 = Model([c0_1, c2_1], [], logits, 'test1', None, None, None)
        model2 = Model([c0_2], [], logits, 'test2', None, None, None)

        w1 = np.ones((6, 6, 2, 2))
        w2 = np.ones((6, 6, 2, 3)) * 2

        b1 = np.zeros(2)
        b2 = np.zeros(3) * 2

        expected_w = np.concatenate((w1, w2), axis=3)
        expected_b = np.concatenate((b1, b2), axis=0)

        ml1 = ModelLayerParameters(w1, b1, None)
        ml2 = ModelLayerParameters(w2, b2, None)

        mp1 = TrainingParameters([], None, None, {'c0': None, 'c1': ml1})
        mp2 = TrainingParameters([], None, None, {'c0': ml2})

        c0_expected = ConvolutionalLayer(5, 3, [20, 20, 3], 'c0')
        c1_expected = ConvolutionalLayer(6, 5, [40, 40, 3], 'c1')
        mutation_params = {'layer_idxs': (1,0)}

        actual_model, actual_mp = cross_models((model1, mp1),(model2, mp2),  **mutation_params)

        expected_model = Model([c0_expected, c1_expected], [], logits, 'test1', ('test1', 'test2'), None, None)

        actual_ml = actual_mp.saved_parameters['c1']
        self.assertTrue((actual_ml.weights==expected_w).all())
        self.assertTrue((actual_ml.biases== expected_b).all())
        self.assertEquals((6, 6, 2, 5), actual_ml.weights.shape)
        self.assertEquals(actual_model, expected_model)