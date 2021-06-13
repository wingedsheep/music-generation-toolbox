import binascii

from mgt.datamanagers.data_manager import Dictionary


class DictionaryGenerator(object):

    @staticmethod
    def create_dictionary() -> Dictionary:
        """
        Creates a dictionary for a REMI mapping of a midi file, with added instruments.
        """

        dictionary = [{}, {}]

        def append_to_dictionary(word):
            if word not in dictionary[0]:
                offset = len(dictionary[0])
                dictionary[0].update({word: offset})
                dictionary[1].update({offset: word})

        def to_hex(short):
            return hex(short)[2:].rjust(2, '0')

        # First word is reserved for padding
        append_to_dictionary("pad")
        append_to_dictionary("mask")

        for i in range(256):
            append_to_dictionary(to_hex(i))

        return Dictionary(dictionary[0], dictionary[1])
