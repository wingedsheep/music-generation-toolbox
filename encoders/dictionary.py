class Dictionary(object):

    def __init__(self, word_to_data, data_to_word):
        self.wtd = word_to_data
        self.dtw = data_to_word

    def word_to_data(self, word):
        return self.wtd[word]

    def data_to_word(self, data):
        return self.dtw[data]

    def size(self):
        return len(self.wtd)
