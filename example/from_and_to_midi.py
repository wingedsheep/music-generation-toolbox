from mgt.datamanagers.remi_data_manager import RemiDataManager
import os

from mgt.datamanagers.time_shift_data_manager import TimeShiftDataManager

"""
Example showing how to parse midi files with different encodings.
Also outputs midi files showing how these encodings sound when converted back.
"""
midi_path = '../data/TheWeeknd-BlindingLights.midi'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)

# Parse midi using timeshift method
timeshift_data_manager = TimeShiftDataManager()
timeshift_dataset = timeshift_data_manager.prepare_data([midi_path])
timeshift_midi = timeshift_data_manager.to_midi(timeshift_dataset.data[0])
timeshift_midi.save("timeshift.midi")

# Parse midi using remi method
remi_data_manager = RemiDataManager()
remi_dataset = remi_data_manager.prepare_data([midi_path])
remi_midi = remi_data_manager.to_midi(remi_dataset.data[0])
remi_midi.save("remi.midi")
