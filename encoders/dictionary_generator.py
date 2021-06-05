from encoders.dictionary import Dictionary


class DictionaryGenerator(object):

    @staticmethod
    def create_dictionary() -> Dictionary:
        """
        Creates a dictionary for a REMI-like mapping of midi events.
        """

        dictionary = [{}, {}]

        def append_to_dictionary(word):
            if word not in dictionary[0]:
                offset = len(dictionary[0])
                dictionary[0].update({word: offset})
                dictionary[1].update({offset: word})

        # First word is reserved for padding
        append_to_dictionary("padding")

        append_to_dictionary("start-track")
        append_to_dictionary("end-track")

        # Instrument indicates the midi instrument program value 0-127
        # and value 128 reserved for instruments with is_drum = true
        for i in range(129):
            append_to_dictionary(f"program_{i}")

        # Midi pitch value between 0-127
        for i in range(128):
            append_to_dictionary(f"note_{i}")

        # Duration indicates the duration of a note in 1/32th note intervals (1-128)
        for i in range(128):
            append_to_dictionary(f"duration_{i + 1}")

        # Time shift in 1/32th note intervals (1-128)
        for i in range(128):
            append_to_dictionary(f"time-shift_{i + 1}")

        # Velocity is a value between 0-127, which we divide into 32 bins
        for i in range(32):
            append_to_dictionary(f"velocity_{i}")

        # Tempo is a value between 10-200 divided into bins of 5 (so 1-40)
        # for i in range(20):
        #     append_to_dictionary(f"tempo_{i + 1}")

        return Dictionary(dictionary[0], dictionary[1])
