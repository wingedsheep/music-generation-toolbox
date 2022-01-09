import re

from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi_data_manager import RemiDataManager

from pathlib import Path

"""
Example showing how to save and load a large dataset that doesn't fit entirely in memory.
"""

parent_midi_path = "MIDI_PATH"
output_path = "OUTPUT_PATH"

midi_paths = []
for path in Path(parent_midi_path).rglob('*.mid'):
    resolved_path = str(path.absolute().resolve())
    duplicate = re.search("^.*[\d]+.mid.*$", resolved_path)
    if not duplicate:
        midi_paths.append(resolved_path)

midi_paths.sort()

data_manager = RemiDataManager()

data = []

songs_per_file = 500
nr_of_saved_files = 0

for midi_path in midi_paths:
    split_path = midi_path.split('/')
    filename = split_path[len(split_path) - 2]

    dataset = data_manager.prepare_data([midi_path])
    data.extend(dataset.data)

    if len(data) >= songs_per_file:
        print(f'Saving file: {nr_of_saved_files}')
        DataHelper.save(data, f'output_path/data_{nr_of_saved_files}')
        nr_of_saved_files += 1
        data = []

DataHelper.save(data, f'output_path/data_{nr_of_saved_files}')
