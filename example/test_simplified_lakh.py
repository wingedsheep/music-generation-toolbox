import os

from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi.instrument_mappings import simplified_instruments
from mgt.datamanagers.remi_data_manager import RemiDataManager

midi_path = '../data/TheWeeknd-BlindingLights.midi'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)

# Parse midi using remi method
remi_data_manager = RemiDataManager(instrument_mapping=simplified_instruments)
# remi_dataset = remi_data_manager.prepare_data([midi_path])
# remi_midi = remi_data_manager.to_midi(remi_dataset.data[0])
# print(remi_midi.midi.instruments)
# remi_midi.save("remi.midi")

data = DataHelper.load("/Users/vincentbons/Documents/Music toolbox/lakh_simplified_transposed_2/lakh_remi_simple_instruments_0")
remi_midi = remi_data_manager.to_midi(data[95])
print(remi_midi.midi.tracks)
remi_midi.save("remi.midi")