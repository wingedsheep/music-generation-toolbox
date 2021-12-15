from pretty_midi import pretty_midi

from mgt.datamanagers.encoded_beats.beat_data_extractor import BeatDataExtractor
from mgt.datamanagers.encoded_beats.encoded_beats_dataset import EncodedBeatsDataSet
from mgt.datamanagers.midi_wrapper import PrettyMidiWrapper

defaults = {
    'tracks': [
        27,  # Electric guitar
        70,  # Bassoon
        33,  # Electric bass
        128  # Drums
    ],
    'beat_resolution': 4
}


class EncodedBeatsDataManager(object):

    def __init__(
            self,
            tracks=defaults['tracks'],
            beat_resolution=defaults['beat_resolution']
    ):
        self.tracks = tracks
        self.beat_resolution = beat_resolution
        self.data_extractor = BeatDataExtractor(
            tracks=tracks,
            beat_resolution=beat_resolution
        )

    def prepare_data(self, midi_paths) -> EncodedBeatsDataSet:
        training_data = []
        for path in midi_paths:
            print(f"Parsing {path}")
            # try:
            midi_data = pretty_midi.PrettyMIDI(path)
            data = self.data_extractor.extract_beats(midi_data)
            training_data.append(data)
            # except Exception as e:
            #     print(e)

        return EncodedBeatsDataSet(
            data=training_data,
            tracks=self.tracks,
            beat_resolution=self.beat_resolution
        )

    def to_midi(self, model_output) -> PrettyMidiWrapper:
        return PrettyMidiWrapper(self.data_extractor.restore_midi(model_output))
