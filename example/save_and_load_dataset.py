import glob
import os

from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi_data_manager import RemiDataManager


"""
Example showing how to save and load a created dataset.
Saving can be done in steps for large datasets that take a lot of time to parse.
"""

midi_path = '../data/pop/'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

datamanager = RemiDataManager()
dataset = datamanager.prepare_data(midis)
DataHelper.save(dataset, 'test_dataset')

loaded_dataset = DataHelper.load('test_dataset')
print(len(loaded_dataset.data))  # Should contain the 10 parsed midis

