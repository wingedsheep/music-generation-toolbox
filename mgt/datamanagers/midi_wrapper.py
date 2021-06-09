import miditoolkit
from pretty_midi import pretty_midi


class MidiWrapper(object):
    def save(self, path):
        pass


class PrettyMidiWrapper(MidiWrapper):

    def __init__(self, midi: pretty_midi.PrettyMIDI):
        self.midi = midi

    def save(self, path):
        self.midi.write(path)


class MidiToolkitWrapper(MidiWrapper):

    def __init__(self, midi: miditoolkit.midi.parser.MidiFile):
        self.midi = midi

    def save(self, path):
        self.midi.dump(path)
