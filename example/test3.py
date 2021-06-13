import glob
import os

from mgt.datamanagers.compressed_remi_data_manager import CompressedRemiDataManager

midi_path = '../data/TheWeeknd-BlindingLights.midi'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)

remi_data_manager = CompressedRemiDataManager()
remi_dataset = remi_data_manager.prepare_data([midi_path])
remi_midi = remi_data_manager.to_midi(remi_dataset.data[0])
remi_midi.save("remi.midi")