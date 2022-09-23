COMPOUND_WORD_PADDING = [0, 0, 0, 0, 0, 0, 0, 0]
COMPOUND_WORD_BAR = [2, 0, 0, 0, 0, 0, 0, 0]


def pad(array: list, max_sequence_length, padding_character=None):
    if padding_character is None:
        padding_character = COMPOUND_WORD_PADDING
    padded_array = array.copy()

    while len(padded_array) < max_sequence_length:
        padded_array.insert(0, padding_character)

    return padded_array
