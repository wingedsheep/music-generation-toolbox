import numpy as np
from typing import List
import random

COMPOUND_WORD_PADDING = [0, 0, 0, 0, 0, 0, 0, 0]
COMPOUND_WORD_BAR = [2, 0, 0, 0, 0, 0, 0, 0]


def pad(array: np.ndarray, max_sequence_length: int, padding_compound_word: np.ndarray = None) -> np.ndarray:
    """
    Pads an array of compound words with a padding compound word to a given length.
    Padding is applied to the end of the array.
    :param array: The array of compound words to pad.
    :param max_sequence_length: The length to pad the array to.
    :param padding_compound_word: The compound word (list of integers) to pad the array with. Default is COMPOUND_WORD_PADDING.
    :return: The padded array.
    """
    if padding_compound_word is None:
        padding_compound_word = COMPOUND_WORD_PADDING

    if len(array) >= max_sequence_length:
        return array[:max_sequence_length]
    else:
        number_of_padding_elements_to_add = max_sequence_length - len(array)
        padding = np.tile(padding_compound_word, (number_of_padding_elements_to_add, 1))
        return np.vstack((array, padding))


def get_batch(training_data: List[np.ndarray], batch_size: int, max_sequence_length: int) -> List[np.ndarray]:
    """
    Creates a batch of random sequences from the training data, each sequence with a specified length.

    :param training_data: A list of sequences representing the training data.
    :param batch_size: The number of sequences in the batch.
    :param max_sequence_length: The length of each sequence in the batch.
    :param randomly_truncate: Whether to randomly truncate sequences in the batch.
    :return: A list of sequences representing the batch.
    """
    indices = []
    for i in range(batch_size):
        song_index = random.randint(0, len(training_data) - 1)
        starting_index = random.randint(0, len(training_data[song_index]) - 1)
        indices.append((song_index, starting_index))

    sequences = []
    for selection in indices:
        song_index = selection[0]
        start_index = selection[1]
        end_index = min(start_index + max_sequence_length, len(training_data[song_index]))
        selection = np.array(training_data[song_index][start_index:end_index])
        padded_selection = pad(selection, max_sequence_length)
        sequences.append(padded_selection)

    return sequences
