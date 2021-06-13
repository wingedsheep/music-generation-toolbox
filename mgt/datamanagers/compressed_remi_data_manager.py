import binascii
import zlib

from pretty_midi import pretty_midi

from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi import util
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.compressed_remi.dictionary_generator import DictionaryGenerator as CompressedDictionaryGenerator


class CompressedRemiDataManager(DataManager):

    def __init__(self, use_chords=True, transposition_steps=None):
        if transposition_steps is None:
            transposition_steps = [0]
        self.use_chords = use_chords
        self.transposition_steps = transposition_steps
        self.dictionary = CompressedDictionaryGenerator.create_dictionary()
        self.dictionary2 = DictionaryGenerator.create_dictionary()

    def prepare_data(self, midi_paths) -> DataSet:
        training_data = []
        for path in midi_paths:
            for transposition_step in self.transposition_steps:
                data = util.extract_data(path,
                                         transposition_steps=transposition_step,
                                         dictionary=self.dictionary,
                                         use_chords=self.use_chords)

                stringified = ' '.join([str(x) for x in data]).encode('ascii')
                compressed = zlib.compress(stringified)
                compressed_hex = binascii.hexlify(compressed)

                words = []
                for i in range(len(compressed_hex), step=2):
                    word = compressed_hex[i] + compressed_hex[i + 1]
                    words.append(self.dictionary.word_to_data(word))

                training_data.append(words)
        return DataSet(training_data, self.dictionary)

    def to_midi(self, data) -> MidiWrapper:
        compressed = ''.join([self.dictionary.data_to_word(x) for x in data])
        decompressed = zlib.decompress(binascii.unhexlify(compressed)).decode('ascii')
        decompressed_data = decompressed.split(" ")
        return MidiToolkitWrapper(util.to_midi(decompressed_data, self.dictionary2))
