import random

import numpy as np
import torch


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad(array, max_sequence_length, padding_character=0):
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


def get_batch(training_data, batch_size, max_sequence_length):
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


def get_long_batch(training_data, batch_size, max_sequence_length, num_segments):
    long_sequences = []

    for _ in range(batch_size):
        song_index = random.randint(0, len(training_data) - 1)
        song_length = len(training_data[song_index])

        segment_length = max_sequence_length * num_segments
        starting_index = random.randint(0, max(0, song_length - 1))

        ending_index = min(song_length, starting_index + segment_length)

        long_sequence = training_data[song_index][starting_index:ending_index]
        padded_long_sequence = pad(long_sequence, segment_length)

        long_sequences.append(padded_long_sequence)

    return long_sequences


def get_or_default(dictionary: dict, key: str, defaults: dict):
    return dictionary[key] if key in dictionary else defaults[key]
