import torch
from pretty_midi import pretty_midi

from encoders.dictionary_generator import DictionaryGenerator
from encoders.event_extractor import EventExtractor
from encoders.midi_generator import MidiGenerator
from encoders.midi_writer import MidiWriter
from encoders.words_converter import WordsConverter
from encoders.input_data_converter import InputDataConverter
import os
import glob

# Load MIDI files
from models.reformer_model import ReformerModel


def run():
    midi_path = '../data/pop/'
    midi_path = os.path.join(os.path.dirname(__file__), midi_path)
    midis = glob.glob(midi_path + '*.mid')

    # Convert MIDI files to training data
    dictionary = DictionaryGenerator.create_dictionary()
    training_data = []
    for midi in midis:
        midi_data = pretty_midi.PrettyMIDI(midi)

        # Extract data from midi and convert to input data
        event_extractor = EventExtractor()
        events = event_extractor.extract_events(midi_data)
        words = WordsConverter.events_to_words(events)
        input_data_converter = InputDataConverter(dictionary)
        input_data = input_data_converter.words_to_input_data(words)
        training_data.append(torch.tensor(input_data))

    print(training_data)

    # Create and train the model
    model = ReformerModel(dictionary)
    model.train(x_train=training_data, epochs=100, stop_loss=0.1)

    # Generate music
    output = model.generate(30)

    # Restore events from input data
    input_data_converter = InputDataConverter(dictionary)
    restored_words = input_data_converter.input_data_to_words(output)
    restored_events = WordsConverter.words_to_events(restored_words)

    # Restore midi from restored events
    midi_generator = MidiGenerator()
    restored_midi = midi_generator.events_to_midi(restored_events)
    MidiWriter.write_midi(restored_midi, "test.midi")

run()
