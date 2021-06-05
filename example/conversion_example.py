from pretty_midi import pretty_midi

from encoders.dictionary_generator import DictionaryGenerator
from encoders.event_extractor import EventExtractor
from encoders.midi_generator import MidiGenerator
from encoders.midi_writer import MidiWriter
from encoders.words_converter import WordsConverter
from encoders.input_data_converter import InputDataConverter
import os

# Load MIDI
midi_path = '../data/TheWeeknd-BlindingLights.midi'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midi_data = pretty_midi.PrettyMIDI(midi_path)

# Extract data from midi and convert to input data
event_extractor = EventExtractor()
events = event_extractor.extract_events(midi_data)
words = WordsConverter.events_to_words(events)
dictionary = DictionaryGenerator.create_dictionary()
input_data_converter = InputDataConverter(dictionary)
input_data = input_data_converter.words_to_input_data(words)

# Restore events from input data
restored_words = input_data_converter.input_data_to_words(input_data)
restored_events = WordsConverter.words_to_events(restored_words)

# Restore midi from restored events
midi_generator = MidiGenerator()
restored_midi = midi_generator.events_to_midi(restored_events)
MidiWriter.write_midi(restored_midi, "test.midi")
