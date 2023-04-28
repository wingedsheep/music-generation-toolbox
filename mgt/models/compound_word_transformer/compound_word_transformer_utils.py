import numpy as np
from typing import List

COMPOUND_WORD_PADDING = [0, 0, 0, 0, 0, 0, 0, 0]
COMPOUND_WORD_BAR = [2, 0, 0, 0, 0, 0, 0, 0]


def pad(array: np.ndarray, max_sequence_length: int, padding_compound_word: List[int] = None) -> np.ndarray:
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
