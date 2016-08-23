from unittest import TestCase
import numpy as np
from ga_nn.class_defs import  *

class TestPossible_cross_overs(TestCase):
    def test_possible_cross_overs_same_langth(self):
        from ga_nn.tournament import possible_cross_overs
        s1 = ModelSummary(name='1_1', filters=np.array([2, 4, 5]), input_channels=np.array([6, 7, 8]))
        s2 = ModelSummary(name='1_2', filters=np.array([3, 5, 3]), input_channels=np.array([1, 8, 3]))
        s3 = ModelSummary(name='1_3', filters=np.array([3, 4, 1]), input_channels=np.array([3, 7, 1]))

        summaries = {'1_1':s1, '1_2':s2, '1_3':s3}

        actual = possible_cross_overs(summaries)
        expected = {('1_1', '1_2'):[(2, 1)], ('1_2', '1_1'):[(1, 2)],
                    ('1_1', '1_3'): [(1, 1)], ('1_3', '1_1'): [(1, 1)],
                    ('1_2', '1_3'): [(2, 0)], ('1_3', '1_2'): [(0, 2)]
                    }

        self.assertDictEqual(actual, expected)


def test_possible_cross_overs_different_langth(self):
    from ga_nn.tournament import possible_cross_overs
    s1 = ModelSummary(name='1_1', filters=np.array([2, 4, 1, 7, 3]), input_channels=np.array([6, 7, 8, 2, 1]))
    s2 = ModelSummary(name='1_2', filters=np.array([3, 5, 3]), input_channels=np.array([1, 8, 3]))

    summaries = {'1_1': s1, '1_2': s2}

    actual = possible_cross_overs(summaries)
    expected = {('1_1', '1_2'): [(4, 2)], ('1_2', '1_1'): [(2, 4)]}

    self.assertDictEqual(actual, expected)


def test_possible_cross_overs_multiple_matches(self):
    from ga_nn.tournament import possible_cross_overs
    s1 = ModelSummary(name='1_1', filters=np.array([3, 5, 1]), input_channels=np.array([1, 8, 3]))
    s2 = ModelSummary(name='1_2', filters=np.array([3, 5, 3]), input_channels=np.array([1, 8, 3]))

    summaries = {'1_1': s1, '1_2': s2}

    actual = possible_cross_overs(summaries)
    expected = {('1_1', '1_2'): [(0, 0), (1, 1)], ('1_2', '1_1'): [(0, 0), (1, 1)]}

    self.assertItemsEqual(actual.keys(), expected.keys())
    for k in expected.keys():
        self.assertItemsEqual(actual[k], expected[k])

