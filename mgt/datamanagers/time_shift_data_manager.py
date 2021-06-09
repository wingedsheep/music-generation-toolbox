from pretty_midi import pretty_midi

from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import PrettyMidiWrapper, MidiWrapper
from mgt.datamanagers.time_shift.input_data_converter import InputDataConverter
from mgt.datamanagers.time_shift.midi_generator import MidiGenerator
from mgt.datamanagers.time_shift.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.time_shift.event_extractor import EventExtractor
from mgt.datamanagers.time_shift.words_converter import WordsConverter


class TimeShiftDataManager(DataManager):

    def __init__(self):
        self.dictionary = DictionaryGenerator.create_dictionary()
        self.event_extractor = EventExtractor()
        self.midi_generator = MidiGenerator()
        self.input_data_converter = InputDataConverter(self.dictionary)

    def prepare_data(self, midi_paths) -> DataSet:
        training_data = []
        for midi in midi_paths:
            print(f"preparing data for {midi}")
            midi_data = pretty_midi.PrettyMIDI(midi)
            events = self.event_extractor.extract_events(midi_data)
            words = WordsConverter.events_to_words(events)
            input_data = self.input_data_converter.words_to_input_data(words)
            training_data.append(input_data)

        return DataSet(training_data, self.dictionary)

    def to_midi(self, data) -> MidiWrapper:
        restored_events = self.to_events(data)
        return PrettyMidiWrapper(self.midi_generator.events_to_midi(restored_events))

    def to_events(self, data):
        restored_words = self.input_data_converter.input_data_to_words(data)
        return WordsConverter.words_to_events(restored_words)
