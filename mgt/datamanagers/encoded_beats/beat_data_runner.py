from pretty_midi import pretty_midi

from mgt.datamanagers.encoded_beats.beat_data_extractor import BeatDataExtractor
from mgt.models.encoded_beats_model import EncodedBeatsModel

data_extractor = BeatDataExtractor()
midi_path = '/Users/vincentbons/Downloads/californication_4_tracks.mid'
midi_data = pretty_midi.PrettyMIDI(midi_path)
beats_matrices = data_extractor.extract_beats(midi_data)

encoded_beats_model = EncodedBeatsModel()
encoded_beats_model.auto_encoder.train(beats_matrices, 1)
encoded_beats_model.train([beats_matrices], epochs=1)
result = encoded_beats_model.generate()
print(result.toarray)
