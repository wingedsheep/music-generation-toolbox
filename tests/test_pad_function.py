import unittest
import numpy as np

from mgt.models.utils import pad


class TestPadFunction(unittest.TestCase):

    def test_no_padding_needed(self):
        input_array = np.array([1, 2, 3])
        max_sequence_length = 3
        expected_output = np.array([1, 2, 3])
        output = pad(input_array, max_sequence_length)
        np.testing.assert_array_equal(output, expected_output)

    def test_truncate_array(self):
        input_array = np.array([1, 2, 3, 4, 5])
        max_sequence_length = 3
        expected_output = np.array([1, 2, 3])
        output = pad(input_array, max_sequence_length)
        np.testing.assert_array_equal(output, expected_output)

    def test_pad_array(self):
        input_array = np.array([1, 2, 3])
        max_sequence_length = 5
        padding_character = 0
        expected_output = np.array([1, 2, 3, 0, 0])
        output = pad(input_array, max_sequence_length, padding_character)
        np.testing.assert_array_equal(output, expected_output)

    def test_pad_array_with_custom_padding_character(self):
        input_array = np.array([1, 2, 3])
        max_sequence_length = 5
        padding_character = -1
        expected_output = np.array([1, 2, 3, -1, -1])
        output = pad(input_array, max_sequence_length, padding_character)
        np.testing.assert_array_equal(output, expected_output)


if __name__ == '__main__':
    unittest.main()
