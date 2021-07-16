from chord_extractor.extractors import Chordino

chordino = Chordino(roll_on=1)

# Optional, only if we need to extract from a file that isn't accepted by librosa
conversion_file_path = chordino.preprocess('/some_path/some_song.mid')

chords = chordino.extract(conversion_file_path)