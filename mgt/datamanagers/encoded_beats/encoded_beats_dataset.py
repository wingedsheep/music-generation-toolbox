from mgt.datamanagers.encoded_beats.encoded_beats_constants import POSSIBLE_MIDI_PITCHES


class EncodedBeatsDataSet(object):

    def __init__(
            self,
            data,
            instruments,
            beat_resolution
    ):
        self.data = data
        self.instruments = instruments
        self.beat_resolution = beat_resolution

    def get_encoded_beat_size(self):
        return len(self.instruments) * self.beat_resolution * POSSIBLE_MIDI_PITCHES
