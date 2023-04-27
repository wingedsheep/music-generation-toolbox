import random
from typing import Dict, List, TypeVar

import numpy as np
import torch

TKey = TypeVar('TKey')
TValue = TypeVar('TValue')


def get_device() -> torch.device:
    """
    Returns the appropriate device for computation (GPU or CPU) based on availability.

    :return: A torch.device object representing the appropriate device.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad(array: np.ndarray, max_sequence_length: int, padding_character: int = 0) -> np.ndarray:
    """
    Pads an array with a padding character to a given length.
    Padding is applied to the end of the array.
    :param array: The array to pad.
    :param max_sequence_length: The length to pad the array to.
    :param padding_character: The character to pad the array with.
    :return: The padded array.
    """
    if len(array) >= max_sequence_length:
        return array[:max_sequence_length]
    else:
        number_of_padding_tokens_to_add = max_sequence_length - len(array)
        return np.pad(array, (0, number_of_padding_tokens_to_add), 'constant', constant_values=padding_character)


def get_batch(training_data: List[np.ndarray], batch_size: int, max_sequence_length: int) -> List[np.ndarray]:
    """
    Creates a batch of random sequences from the training data, each sequence with a specified length.

    :param training_data: A list of sequences representing the training data.
    :param batch_size: The number of sequences in the batch.
    :param max_sequence_length: The length of each sequence in the batch.
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
        selection = training_data[song_index][start_index:end_index]
        padded_selection = pad(selection, max_sequence_length)
        sequences.append(padded_selection)

    return sequences


def get_or_default(dictionary: Dict[TKey, TValue], key: TKey, defaults: Dict[TKey, TValue]) -> TValue:
    """
    Returns the value for the specified key in the dictionary if it exists,
    otherwise returns the default value for the key.

    :param dictionary: A dictionary containing key-value pairs.
    :param key: The key to look for in the dictionary.
    :param defaults: A dictionary containing default values for keys.
    :return: The value for the specified key, or the default value if the key is not in the dictionary.
    """
    return dictionary[key] if key in dictionary else defaults[key]
