from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi_data_manager import RemiDataManager

from pathlib import Path

midi_paths = []

for path in Path('/Users/vincentbons/Downloads/clean_midi').rglob('*.mid'):
    midi_paths.append(str(path.absolute().resolve()))

midi_paths.sort()

datamanager = RemiDataManager(transposition_steps=[-1, 0, 1])

process = False

character = '0'
start_from_character = 't'
data = []

for midi_path in midi_paths:
    split_path = midi_path.split('/')
    filename = split_path[len(split_path) - 2]
    if not filename[0].isalpha():
        current_character = '0'
    else:
        current_character = filename[0].lower()

    print(f'Current character: {current_character}')

    if current_character < start_from_character:
        print(f'Skipping {current_character}')
        continue

    if current_character != character:
        DataHelper.save(data, f'/Users/vincentbons/Documents/Music toolbox/lakh_remi_{character}')
        data = []
        character = current_character

    dataset = datamanager.prepare_data([midi_path])
    data.extend(dataset.data)
    print(len(data))

DataHelper.save(data, f'/Users/vincentbons/Documents/Music toolbox/lakh_remi_{character}')
