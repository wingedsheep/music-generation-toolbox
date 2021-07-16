from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi_data_manager import RemiDataManager

midis = []

for i in range(1, 910):
    dir = str(i).zfill(3)
    midis.append(
        '/Users/vincentbons/Documents/AI Song contest/pop909/POP909-Dataset/POP909/' + dir + '/' + dir + '.mid')

midis.sort()

datamanager = RemiDataManager(transposition_steps=[-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
                              map_tracks_to_instruments={0: 0, 1: 1, 2: 2})

dataset = datamanager.prepare_data(midis)

DataHelper.save(dataset.data, f'/Users/vincentbons/Documents/Music toolbox/pop909_split_instruments')
