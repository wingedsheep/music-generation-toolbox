import pickle

from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.data_manager import DataSet

dictionary = None
data = []
with open('/Users/vincentbons/Documents/Music toolbox/lakh_remi_from_bee_gees_how_can_you_mend_a_broken_heart', "rb") as f:
    while True:
        try:
            dataset = pickle.load(f)
            dictionary = dataset.dictionary
            data.append(pickle.load(f).data)
        except:
            break

dataset = DataSet(data=data, dictionary=dictionary)
DataHelper.save(dataset, '/Users/vincentbons/Documents/Music toolbox/lakh_remi_dataset_from_bee_gees')
