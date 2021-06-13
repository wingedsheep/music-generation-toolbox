from mgt.datamanagers.remi import util
from mgt.datamanagers.remi_data_manager import RemiDataManager
import zlib
import binascii

path = "/Users/vincentbons/Documents/AI Song contest/midi/rhcp/Red Hot Chili Peppers - Dani California.mid"

remi_data_manager = RemiDataManager()
remi_dataset = remi_data_manager.prepare_data([path])
print(remi_dataset.data[0])

tuple_count = {}
for i in len(remi_dataset.data[0]):