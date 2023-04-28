from unittest.mock import patch

import numpy as np
import unittest

from mgt.models.compound_word_transformer.compound_word_transformer_utils import pad, get_batch


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


class TestGetBatchFunction(unittest.TestCase):

    def setUp(self):
        self.training_data = [
            np.array([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 0, 0, 0, 0],
                [3, 0, 0, 0, 0, 0, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0]
            ]),
            np.array([
                [5, 0, 0, 0, 0, 0, 0, 0],
                [6, 0, 0, 0, 0, 0, 0, 0],
                [7, 0, 0, 0, 0, 0, 0, 0]
            ])
        ]

    def test_batch_size(self):
        batch_size = 5
        max_sequence_length = 3

        batch = get_batch(self.training_data, batch_size, max_sequence_length)
        print(batch)
        self.assertEqual(len(batch), batch_size)

    def test_max_sequence_length(self):
        batch_size = 5
        max_sequence_length = 3

        batch = get_batch(self.training_data, batch_size, max_sequence_length)
        for sequence in batch:
            self.assertEqual(len(sequence), max_sequence_length)


if __name__ == '__main__':
    unittest.main()
