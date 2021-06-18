import pickle


class DataHelper(object):

    @staticmethod
    def save(data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return pickle.load(f)
