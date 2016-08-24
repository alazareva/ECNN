from unittest import TestCase
from ecnn.class_defs import  *


class Test_model_to_string(TestCase):

    def test_model_to_string(self):
        from ecnn.tournament import model_to_string
        c0 = ConvolutionalLayer(name='c0', filter_size=10, filters=5)
        c1 = ConvolutionalLayer(name='c1', filter_size=2, filters=1)
        d0 = DenseLayer(name='d0', hidden_units=600)

        model = Model()
        model.convolutional_layers.append(c0)
        model.convolutional_layers.append(c1)
        model.dense_layers.append(d0)
        actual = model_to_string(model)
        expected = '[c0, fs: 10, f: 5] --> [c1, fs: 2, f: 1] --> [d0, h: 600] --> logits'
        self.assertEqual(actual,expected)
