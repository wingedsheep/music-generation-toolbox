from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi_data_manager import RemiDataManager

from pathlib import Path

midi_paths = []

for path in Path('/Users/vincentbons/Downloads/clean_midi').rglob('*.mid'):
    midi_paths.append(str(path.absolute().resolve()))

midi_paths.sort()

datamanager = RemiDataManager(transposition_steps=[-2, -1, 0, 1, 2])

process = False

character = '0'
data = []

i = 0
for midi_path in midi_paths:
    if not midi_path[0].isalpha():
        current_character = '0'
    else:
        current_character = midi_path[0].lower()

    print(current_character)

    if i == 3:
        DataHelper.save(data, f'/Users/vincentbons/Documents/Music toolbox/lakh_remi_{character}')
        data = []
        character = current_character
        break

    dataset = datamanager.prepare_data([midi_path])
    data.extend(dataset.data)

    i += 1

print(len(DataHelper.load('/Users/vincentbons/Documents/Music toolbox/lakh_remi_0')))
