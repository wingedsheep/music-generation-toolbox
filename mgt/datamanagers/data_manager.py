from mgt.datamanagers.midi_wrapper import MidiWrapper


class Dictionary(object):

    def __init__(self, word_to_data, data_to_word):
        self.wtd = word_to_data
        self.dtw = data_to_word

    def word_to_data(self, word):
        try:
            return self.wtd[word]
        except Exception as e:
            print(f"{word} not in dictionary)")

    def data_to_word(self, data):
        return self.dtw[data]

    def size(self):
        return len(self.wtd)


class DataSet(object):

    def __init__(self, data, dictionary: Dictionary):
        self.data = data
        self.dictionary = dictionary


class DataManager(object):

    def prepare_data(self, midi_paths) -> DataSet:
        pass

    def to_midi(self, data) -> MidiWrapper:
        pass
