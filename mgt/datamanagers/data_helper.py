import pickle
import os

class DataHelper(object):

    @staticmethod
    def save(data, path):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def extend(data, path):
        if not os.path.exists(path):
            DataHelper.save(data, path)
        else:
            with open(path, 'ab+') as f:
                pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
