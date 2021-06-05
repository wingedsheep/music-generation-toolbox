import pretty_midi

from midi_converter.event_extractor import Event


class MidiWriter(object):

    @classmethod
    def write_midi(cls, midi: pretty_midi.PrettyMIDI, path):
        midi.write(path)
