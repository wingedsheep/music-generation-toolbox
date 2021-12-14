from mgt.datamanagers.encoded_beats.beat_data_extractor import BeatDataExtractor
from mgt.datamanagers.encoded_beats.encoded_beats_dataset import EncodedBeatsDataSet
from mgt.datamanagers.midi_wrapper import PrettyMidiWrapper

defaults = {
    'instruments': {
        27,  # Electric guitar
        70,  # Bassoon
        33,  # Electric bass
        128  # Drums
    },
    'beat_resolution': 4
}


class EncodedBeatsDataManager(object):

    def __init__(
            self,
            instruments=defaults['instruments'],
            beat_resolution=defaults['beat_resolution']
    ):
        self.instruments = instruments
        self.beat_resolution = beat_resolution
        self.data_extractor = BeatDataExtractor(
            instruments=instruments,
            beat_resolution=beat_resolution
        )

    def prepare_data(self, midi_paths) -> EncodedBeatsDataSet:
        training_data = []
        for path in midi_paths:
            try:
                data = self.data_extractor.extract_beats(path)
                training_data.append(data)
            except Exception as e:
                print(e)

        return EncodedBeatsDataSet(
            data=training_data,
            instruments=self.instruments,
            beat_resolution=self.beat_resolution
        )

    def to_midi(self, model_output) -> PrettyMidiWrapper:
        return PrettyMidiWrapper(self.data_extractor.restore_midi(model_output))
