from mgt.datamanagers.data_manager import Dictionary


class DictionaryGenerator(object):

    @staticmethod
    def create_dictionary() -> Dictionary:
        """
        Creates a dictionary for a REMI-like mapping of midi events.
        """

        dictionary = Dictionary()

        # First word is reserved for padding
        dictionary.append("pad")
        dictionary.append("mask")

        dictionary.append("start-track")
        dictionary.append("end-track")

        # Instrument indicates the midi instrument program value 0-127
        # and value 128 reserved for instruments with is_drum = true
        for i in range(129):
            dictionary.append(f"program_{i}")

        # Midi pitch value between 0-127
        for i in range(128):
            dictionary.append(f"note_{i}")

        # Duration indicates the duration of a note in 1/32th note intervals (1-128)
        for i in range(128):
            dictionary.append(f"duration_{i + 1}")

        # Time shift in 1/32th note intervals (1-128)
        for i in range(128):
            dictionary.append(f"time-shift_{i + 1}")

        # Velocity is a value between 0-127, which we divide into 32 bins
        for i in range(32):
            dictionary.append(f"velocity_{i}")

        # Tempo is a value between 10-200 divided into bins of 5 (so 1-40)
        # for i in range(20):
        #     append_to_dictionary(f"tempo_{i + 1}")

        return dictionary
