from mgt.datamanagers.remi import util
from mgt.datamanagers.remi_data_manager import RemiDataManager
import zlib
import binascii

path = "/Users/vincentbons/Documents/AI Song contest/midi/rhcp/Red Hot Chili Peppers - Dani California.mid"

remi_data_manager = RemiDataManager()
remi_dataset = remi_data_manager.prepare_data([path])
stringified = ' '.join([str(x) for x in remi_dataset.data[0]]).encode('ascii')
compressed = zlib.compress(stringified)
compressed_hex = binascii.hexlify(compressed)
decompressed = zlib.decompress(binascii.unhexlify(compressed_hex))

print(len(binascii.hexlify(stringified)))
print(len(compressed_hex))
print(len(binascii.hexlify(decompressed)))

print(compressed_hex[0:20])
