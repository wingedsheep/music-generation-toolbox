import unittest
import numpy as np

from mgt.models.utils import get_batch


class TestGetBatch(unittest.TestCase):

    def setUp(self):
        self.training_data = [
            np.random.randint(0, 50, size=(5,)),
            np.random.randint(0, 50, size=(6,)),
        ]

    def test_batch_size(self):
        batch_size = 3
        max_sequence_length = 4
        batch = get_batch(self.training_data, batch_size, max_sequence_length)
        self.assertEqual(len(batch), batch_size, "Batch size should be equal to the requested batch size")

    def test_sequence_length(self):
        batch_size = 3
        max_sequence_length = 4
        batch = get_batch(self.training_data, batch_size, max_sequence_length)
        for sequence in batch:
            self.assertEqual(len(sequence), max_sequence_length,
                             "Sequence length should be equal to the requested max sequence length")

    def test_padded_sequences(self):
        batch_size = 2
        max_sequence_length = 7
        padding_character = 0
        batch = get_batch(self.training_data, batch_size, max_sequence_length)
        for sequence in batch:
            padding_count = np.count_nonzero(sequence == padding_character)
            self.assertGreaterEqual(padding_count, 0,
                                    "There should be padding when the original sequence is shorter than max_sequence_length")


if __name__ == '__main__':
    unittest.main()
