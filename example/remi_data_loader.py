from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.datamanagers.data_helper import DataHelper

from pathlib import Path

midi_paths = []

for path in Path('/Users/vincentbons/Documents/AI Song contest/pop909/POP909-Dataset/POP909').rglob('*.midi'):
    midi_paths.append(str(path.absolute().resolve()))

for path in Path('/Users/vincentbons/Documents/AI Song contest/pop909/POP909-Dataset/POP909').rglob('*.mid'):
    midi_paths.append(str(path.absolute().resolve()))

midi_paths = list(filter(lambda x: 'versions' not in x, midi_paths))

print(f"nr of midis: {len(midi_paths)}")

midi_paths.sort()

datamanager = RemiDataManager(transposition_steps=[-2, -1, 0, 1, 2])
data = []
dataset = datamanager.prepare_data(midi_paths)
data.extend(dataset.data)
DataHelper.save(data, f'/Users/vincentbons/Documents/Music toolbox/pop909_remi_transposed2')
