import string

from midi_converter.dictionary import Dictionary


class InputDataConverter(object):

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def words_to_input_data(self, words: [string]):
        return list(map(lambda x: self.dictionary.word_to_data(x), words))

    def input_data_to_words(self, input_data: [int]):
        return list(map(lambda x: self.dictionary.data_to_word(x), input_data))
