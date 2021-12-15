from mgt.datamanagers.encoded_beats.encoded_beats_constants import POSSIBLE_MIDI_PITCHES


class EncodedBeatsDataSet(object):

    def __init__(
            self,
            data,
            tracks,
            beat_resolution
    ):
        self.data = data
        self.tracks = tracks
        self.beat_resolution = beat_resolution

    def get_encoded_beat_size(self):
        return len(self.tracks) * self.beat_resolution * POSSIBLE_MIDI_PITCHES
