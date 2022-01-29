from mgt.datamanagers.data_manager import DataManager, DataSet
from mgt.datamanagers.midi_wrapper import MidiWrapper, MidiToolkitWrapper
from mgt.datamanagers.remi.data_extractor import DataExtractor
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig
from mgt.datamanagers.remi.efficient_remi_converter import EfficientRemiConverter
from mgt.datamanagers.remi.to_midi_mapper import ToMidiMapper


defaults = {
    'use_chords': False,
    'transposition_steps': [0],
    'map_tracks_to_instruments': {},
    'instrument_mapping': {},
    'efficient_remi_config': EfficientRemiConfig()
}


class RemiDataManager(DataManager):
    """
    use_chords: Should the data manager try to extract chord events based on the played notes.
                This does not work very well for multi instrument midi.
    transposition_steps: Transposed copies of the data to include. For example [-1, 0, 1] has a copy that is transposed
                One semitone down, once the original track, and once transposed one semitone up.
    map_tracks_to_instruments: Whether to map certain track numbers to instruments. For example {0=0, 1=25} maps
                track 0 to a grand piano, and track 1 to an acoustic guitar.
    instrument_mapping: Maps instruments to different instruments. For example {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
                maps all piano-like instruments to a grand piano. Mapping to None removes the instrument entirely.
    efficient_remi: Does not repeat instrument and position for every note if they are the same as the previous.
    """

    def __init__(
            self,
            use_chords=defaults['use_chords'],
            transposition_steps=defaults['transposition_steps'],
            map_tracks_to_instruments=defaults['map_tracks_to_instruments'],
            instrument_mapping=defaults['instrument_mapping'],
            efficient_remi_config=defaults['efficient_remi_config']
    ):
        self.use_chords = use_chords
        self.transposition_steps = transposition_steps
        self.map_tracks_to_instruments = map_tracks_to_instruments
        self.instrument_mapping = instrument_mapping
        self.dictionary = DictionaryGenerator.create_dictionary()
        self.data_extractor = DataExtractor(
            dictionary=self.dictionary,
            map_tracks_to_instruments=self.map_tracks_to_instruments,
            use_chords=self.use_chords,
            instrument_mapping=self.instrument_mapping
        )
        self.efficient_remi_config = efficient_remi_config
        if self.efficient_remi_config.enabled:
            self.efficient_remi_converter = EfficientRemiConverter(efficient_remi_config)
        self.to_midi_mapper = ToMidiMapper(self.dictionary)

    def prepare_data(self, midi_paths) -> DataSet:
        training_data = []
        for path in midi_paths:
            for transposition_step in self.transposition_steps:
                try:
                    if self.efficient_remi_config.enabled:
                        events = self.data_extractor.extract_events(path, transposition_step)
                        words = self.efficient_remi_converter.convert_to_efficient_remi(events)
                        data = self.data_extractor.words_to_data(words)
                        print(f"Parsed {len(data)} words from midi as efficient REMI.")
                        training_data.append(data)
                    else:
                        data = self.data_extractor.extract_data(path, transposition_step)
                        training_data.append(data)
                except Exception as e:
                    print(e)

        return DataSet(training_data, self.dictionary)

    def to_midi(self, data) -> MidiWrapper:
        if self.efficient_remi_config.enabled:
            efficient_words = list(map(lambda x: self.dictionary.data_to_word(x), data))
            words = self.efficient_remi_converter.convert_to_normal_remi(efficient_words)
            data = self.data_extractor.words_to_data(words)

        return MidiToolkitWrapper(self.to_midi_mapper.to_midi(data))
