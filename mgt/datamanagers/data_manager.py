from mgt.datamanagers.midi_wrapper import MidiWrapper


class Dictionary(object):

    def __init__(self):
        self.wtd = {}
        self.dtw = {}

    def word_to_data(self, word):
        return self.wtd[word]

    def data_to_word(self, data):
        return self.dtw[data]

    def size(self):
        return len(self.wtd)

    def append(self, word):
        if word not in self.wtd:
            offset = len(self.wtd)
            self.wtd.update({word: offset})
            self.dtw.update({offset: word})


class DataSet(object):

    def __init__(self, data, dictionary: Dictionary):
        self.data = data
        self.dictionary = dictionary


class DataManager(object):

    def prepare_data(self, midi_paths) -> DataSet:
        pass

    def to_midi(self, data) -> MidiWrapper:
        pass
