import numpy as np
import unittest

from mgt.models.compound_word_transformer.compound_word_transformer_utils import pad


class TestPadFunction(unittest.TestCase):

    def test_no_padding_needed(self):
        array = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0, 0]])
        max_sequence_length = 2
        expected_output = array.copy()

        result = pad(array, max_sequence_length)
        np.testing.assert_array_equal(result, expected_output)

    def test_padding_needed(self):
        array = np.array([[2, 0, 0, 0, 0, 0, 0, 0]])
        max_sequence_length = 3
        expected_output = np.array([[2, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0]])

        result = pad(array, max_sequence_length)
        np.testing.assert_array_equal(result, expected_output)

    def test_custom_padding_character(self):
        array = np.array([[2, 0, 0, 0, 0, 0, 0, 0]])
        max_sequence_length = 3
        padding_character = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        expected_output = np.array([[2, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 1, 1, 1, 1, 1, 1, 1],
                                    [1, 1, 1, 1, 1, 1, 1, 1]])

        result = pad(array, max_sequence_length, padding_character)
        np.testing.assert_array_equal(result, expected_output)

    def test_truncate_array(self):
        array = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                          [2, 0, 0, 0, 0, 0, 0, 0],
                          [3, 0, 0, 0, 0, 0, 0, 0]])
        max_sequence_length = 2
        expected_output = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                    [2, 0, 0, 0, 0, 0, 0, 0]])

        result = pad(array, max_sequence_length)
        np.testing.assert_array_equal(result, expected_output)


if __name__ == '__main__':
    unittest.main()
