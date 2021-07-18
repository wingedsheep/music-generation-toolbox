from mgt.datamanagers.data_manager import Dictionary


class DictionaryGenerator(object):

    @staticmethod
    def create_dictionary() -> Dictionary:
        """
        Creates a dictionary for a REMI mapping of a midi file, with added instruments.
        """

        dictionary = Dictionary()

        # First word is reserved for padding
        dictionary.append("pad")
        dictionary.append("mask")

        dictionary.append('Bar_None')

        chords = ['Chord_A#:aug', 'Chord_A#:dim', 'Chord_A#:dom', 'Chord_A#:maj', 'Chord_A#:min', 'Chord_A:aug',
                  'Chord_A:dim', 'Chord_A:dom', 'Chord_A:maj', 'Chord_A:min', 'Chord_B:aug', 'Chord_B:dim',
                  'Chord_B:dom',
                  'Chord_B:maj', 'Chord_B:min', 'Chord_C#:aug', 'Chord_C#:dim', 'Chord_C#:dom', 'Chord_C#:maj',
                  'Chord_C#:min',
                  'Chord_C:aug', 'Chord_C:dim', 'Chord_C:dom', 'Chord_C:maj', 'Chord_C:min', 'Chord_D#:aug',
                  'Chord_D#:dim',
                  'Chord_D#:dom', 'Chord_D#:maj', 'Chord_D#:min', 'Chord_D:aug', 'Chord_D:dim', 'Chord_D:dom',
                  'Chord_D:maj',
                  'Chord_D:min', 'Chord_E:aug', 'Chord_E:dim', 'Chord_E:dom', 'Chord_E:maj', 'Chord_E:min',
                  'Chord_F#:aug',
                  'Chord_F#:dim', 'Chord_F#:dom', 'Chord_F#:maj', 'Chord_F#:min', 'Chord_F:aug', 'Chord_F:dim',
                  'Chord_F:dom',
                  'Chord_F:maj', 'Chord_F:min', 'Chord_G#:aug', 'Chord_G#:dim', 'Chord_G#:dom', 'Chord_G#:maj',
                  'Chord_G#:min',
                  'Chord_G:aug', 'Chord_G:dim', 'Chord_G:dom', 'Chord_G:maj', 'Chord_G:min', 'Chord_N:N']

        tempo_classes = ['Tempo Class_fast', 'Tempo Class_mid', 'Tempo Class_slow']

        for chord in chords:
            dictionary.append(chord)

        for i in range(129):
            dictionary.append(f"Instrument_{i}")

        for i in range(64):
            dictionary.append(f"Note Duration_{i}")

        for i in range(128):
            dictionary.append(f"Note On_{i}")

        for i in range(32):
            dictionary.append(f"Note Velocity_{i}")

        for i in range(16):
            dictionary.append(f"Position_{i + 1}/16")

        for tempo_class in tempo_classes:
            dictionary.append(tempo_class)

        for i in range(64):
            dictionary.append(f"Tempo Value_{i}")

        return dictionary
