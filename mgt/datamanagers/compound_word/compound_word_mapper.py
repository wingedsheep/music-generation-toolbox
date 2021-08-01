import string

from mgt.datamanagers.data_manager import Dictionary

NUMBER_OF_TEMPO_VALUES = 64


class WordType(object):
    PADDING = 0
    EOS = 1
    TIMING = 2
    NOTE = 3


class CompoundWord(object):
    word_type: int
    bar_beat: int
    tempo: int
    instrument: int
    pitch: int
    duration: int
    velocity: int

    def __init__(self, word_type, bar_beat=0, tempo=0, instrument=0, pitch=0, duration=0, velocity=0):
        self.word_type = word_type
        self.bar_beat = bar_beat
        self.tempo = tempo
        self.instrument = instrument
        self.pitch = pitch
        self.duration = duration
        self.velocity = velocity

    def __repr__(self):
        type_string = 'UNKNOWN'
        if self.word_type == 0:
            type_string = 'PADDING'
        elif self.word_type == 1:
            type_string = 'EOS'
        elif self.word_type == 2:
            type_string = 'TIMING'
        elif self.word_type == 3:
            type_string = 'NOTE'

        return f'CompoundWord(type={type_string}, bar_beat={self.bar_beat}, tempo={self.tempo}, instrument={self.instrument}, pitch={self.pitch}, duration={self.duration}, velocity={self.velocity})'


def create_bar_event():
    return CompoundWord(word_type=WordType.TIMING, bar_beat=0)


def create_beat_event(beat, tempo):
    return CompoundWord(word_type=WordType.TIMING, bar_beat=beat + 1, tempo=tempo)


def create_note_event(instrument, pitch, duration, velocity):
    return CompoundWord(
        word_type=WordType.NOTE,
        instrument=instrument,
        pitch=pitch,
        duration=duration,
        velocity=velocity)


def create_eos_event():
    return CompoundWord(word_type=WordType.EOS)


def map_word(key, offset):
    return key - offset


class CompoundWordMapper(object):

    def __init__(self, dictionary):
        self.dictionary = dictionary

        instrument_keys = {k: v for k, v in dictionary.dtw.items() if 'Instrument' in v}.keys()
        self.instrument_size = len(instrument_keys)
        self.instrument_offset = min(instrument_keys)

        tempo_keys = {k: v for k, v in dictionary.dtw.items() if 'Tempo' in v}.keys()
        self.tempo_size = len(tempo_keys)
        self.tempo_offset = min(tempo_keys)

        note_duration_keys = {k: v for k, v in dictionary.dtw.items() if 'Note Duration' in v}.keys()
        self.note_duration_size = len(note_duration_keys)
        self.note_duration_offset = min(note_duration_keys)

        note_velocity_keys = {k: v for k, v in dictionary.dtw.items() if 'Note Velocity' in v}.keys()
        self.note_velocity_size = len(note_velocity_keys)
        self.note_velocity_offset = min(note_velocity_keys)

        position_keys = {k: v for k, v in dictionary.dtw.items() if 'Position' in v}.keys()
        self.position_size = len(position_keys)
        self.position_offset = min(position_keys)

        pitch_keys = {k: v for k, v in dictionary.dtw.items() if 'Note On' in v}.keys()
        self.pitch_size = len(pitch_keys)
        self.pitch_offset = min(pitch_keys)

    def map_to_compound(self, remi_words: [string], dictionary: Dictionary) -> [CompoundWord]:
        compound_words = []
        prev_position = None
        current_tempo = None
        for i in range(len(remi_words)):
            if remi_words[i] == 'Bar_None':
                compound_words.append(create_bar_event())
                prev_position = None
            elif i + 4 < len(remi_words) and \
                    'Position' in remi_words[i] and \
                    'Instrument' in remi_words[i + 1] and \
                    'Note Velocity' in remi_words[i + 2] and \
                    'Note On' in remi_words[i + 3] and \
                    'Note Duration' in remi_words[i + 4]:

                current_position = map_word(dictionary.wtd[remi_words[i]], self.position_offset)
                if prev_position is None or prev_position != current_position:
                    compound_words.append(create_beat_event(current_position, current_tempo))
                    prev_position = current_position

                instrument_position = map_word(dictionary.wtd[remi_words[i + 1]], self.instrument_offset)
                velocity_position = map_word(dictionary.wtd[remi_words[i + 2]], self.note_velocity_offset)
                pitch_position = map_word(dictionary.wtd[remi_words[i + 3]], self.pitch_offset)
                duration_position = map_word(dictionary.wtd[remi_words[i + 4]], self.note_duration_offset)
                compound_words.append(create_note_event(
                    instrument=instrument_position,
                    velocity=velocity_position,
                    pitch=pitch_position,
                    duration=duration_position))
            elif i + 2 < len(remi_words) and \
                    'Position' in remi_words[i] and \
                    'Tempo Class' in remi_words[i + 1] and \
                    'Tempo Value' in remi_words[i + 2]:

                tempo_class = map_word(dictionary.wtd[remi_words[i + 1]], self.tempo_offset)
                tempo_value = map_word(dictionary.wtd[remi_words[i + 2]], self.tempo_offset)
                current_tempo = tempo_class * NUMBER_OF_TEMPO_VALUES + tempo_value
                current_position = map_word(dictionary.wtd[remi_words[i]], self.position_offset)
                if prev_position is None or prev_position != current_position:
                    compound_words.append(create_beat_event(current_position, current_tempo))
                    prev_position = current_position

        compound_words.append(create_eos_event())

        return compound_words

    @staticmethod
    def map_compound_words_to_data(compound_words: [CompoundWord]):
        return list(map(lambda x: [
            x.word_type,
            x.bar_beat,
            x.tempo,
            x.instrument,
            x.pitch,
            x.duration,
            x.velocity
        ], compound_words))

    def map_to_remi(self, compound_data: [[int]]):
        result = []
        current_position = 0
        for compound_word in compound_data:
            remi_sequence, current_position = self.map_compound_word_to_remi(compound_word, current_position)
            result.extend(remi_sequence)
        return result

    def map_compound_word_to_remi(self, compound_word, current_position):
        word_type = compound_word[0]
        if word_type == 2:
            return self.map_compound_note_to_remi(compound_word, current_position), current_position
        elif word_type == 3:
            return self.map_compound_timing_to_remi(compound_word)

    def map_compound_note_to_remi(self, compound_word, current_position):
        position = current_position + self.position_offset
        instrument = compound_word[3] + self.instrument_offset
        pitch = compound_word[4] + self.pitch_offset
        duration = compound_word[5] + self.note_duration_offset
        velocity = compound_word[6] + self.note_velocity_offset
        return [position, instrument, pitch, duration, velocity]

    def map_compound_timing_to_remi(self, compound_word):
        bar_beat = compound_word[1]
        if bar_beat == 0:
            return [self.dictionary.word_to_data('Bar_None')], 0
        else:
            tempo = compound_word[2]
            tempo_class = int(tempo / NUMBER_OF_TEMPO_VALUES) + self.tempo_offset
            tempo_value = tempo % NUMBER_OF_TEMPO_VALUES + 3 + self.tempo_offset
            position = bar_beat - 1 + self.position_offset
            return [position, tempo_class, tempo_value], bar_beat - 1
