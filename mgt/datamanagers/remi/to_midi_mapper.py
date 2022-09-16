import miditoolkit
import numpy as np

from mgt.datamanagers.data_manager import Dictionary
from mgt.datamanagers.remi.constants import DEFAULT_VELOCITY_BINS, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_INTERVALS, \
    DEFAULT_RESOLUTION, DEFAULT_FRACTION, DRUM_INSTRUMENT
from mgt.datamanagers.remi.event import Event


def note_name_and_octave_to_pitch(note_name, octave):
    return octave * 12 + note_name


class ToMidiMapper(object):

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def to_midi(self, data: []):
        words = list(map(lambda x: self.dictionary.data_to_word(x), data))
        events: [Event] = self.words_to_events(words)

        # get downbeat and note (no time)
        temp_notes = []
        temp_chords = []
        temp_tempos = []
        for i in range(len(events) - 3):
            if events[i].name == 'Bar' and i > 0:
                temp_notes.append('Bar')
                temp_chords.append('Bar')
                temp_tempos.append('Bar')
            elif i + 4 < len(events) and events[i].name == 'Position' and \
                    events[i + 1].name == 'Instrument' and \
                    events[i + 2].name == 'Note Velocity' and \
                    events[i + 3].name == 'Note On' and \
                    events[i + 4].name == 'Note Duration':

                # start time and end time from position
                position = int(events[i].value.split('/')[0]) - 1
                # instrument
                instrument = int(events[i + 1].value)
                # velocity
                index = int(events[i + 2].value)
                velocity = int(DEFAULT_VELOCITY_BINS[index])
                # pitch
                pitch = int(events[i + 3].value)
                # duration
                index = int(events[i + 4].value)
                duration = DEFAULT_DURATION_BINS[index]
                # adding
                temp_notes.append([position, velocity, pitch, duration, instrument])
            elif i + 5 < len(events) and events[i].name == 'Position' and \
                    events[i + 1].name == 'Instrument' and \
                    events[i + 2].name == 'Note Velocity' and \
                    events[i + 3].name == 'Note Name' and \
                    events[i + 4].name == 'Octave' and \
                    events[i + 5].name == 'Note Duration':

                # start time and end time from position
                position = int(events[i].value.split('/')[0]) - 1
                # instrument
                instrument = int(events[i + 1].value)
                # velocity
                index = int(events[i + 2].value)
                velocity = int(DEFAULT_VELOCITY_BINS[index])
                # pitch
                note_name = int(events[i + 3].value)
                octave = int(events[i + 4].value)
                pitch = note_name_and_octave_to_pitch(note_name, octave)
                # duration
                index = int(events[i + 4].value)
                duration = DEFAULT_DURATION_BINS[index]
                # adding
                temp_notes.append([position, velocity, pitch, duration, instrument])
            elif events[i].name == 'Position' and events[i + 1].name == 'Chord':
                position = int(events[i].value.split('/')[0]) - 1
                temp_chords.append([position, events[i + 1].value])
            elif i + 2 < len(events) and events[i].name == 'Position' and \
                    events[i + 1].name == 'Tempo Class' and \
                    events[i + 2].name == 'Tempo Value':
                position = int(events[i].value.split('/')[0]) - 1
                if events[i + 1].value == 'slow':
                    tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i + 2].value)
                elif events[i + 1].value == 'mid':
                    tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i + 2].value)
                elif events[i + 1].value == 'fast':
                    tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i + 2].value)
                temp_tempos.append([position, tempo])
        # get specific time for notes
        ticks_per_beat = DEFAULT_RESOLUTION
        ticks_per_bar = DEFAULT_RESOLUTION * 4  # assume 4/4
        notes = {}
        current_bar = 0
        for note in temp_notes:
            if note == 'Bar':
                current_bar += 1
            else:
                position, velocity, pitch, duration, instrument = note
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                # duration (end time)
                et = st + duration
                notes.setdefault(instrument, []).append(miditoolkit.Note(velocity, pitch, st, et))
        # get specific time for chords
        if len(temp_chords) > 0:
            chords = []
            current_bar = 0
            for chord in temp_chords:
                if chord == 'Bar':
                    current_bar += 1
                else:
                    position, value = chord
                    # position (start time)
                    current_bar_st = current_bar * ticks_per_bar
                    current_bar_et = (current_bar + 1) * ticks_per_bar
                    flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                    st = flags[position]
                    chords.append([st, value])
        # get specific time for tempos
        tempos = []
        current_bar = 0
        for tempo in temp_tempos:
            if tempo == 'Bar':
                current_bar += 1
            else:
                position, value = tempo
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                tempos.append([int(st), value])

        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION

        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes

        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))

        for instrument, instrument_notes in notes.items():
            program = 0 if instrument == DRUM_INSTRUMENT else instrument
            is_drum = True if instrument == DRUM_INSTRUMENT else False
            inst = miditoolkit.midi.containers.Instrument(program, is_drum=is_drum)
            inst.notes = instrument_notes
            midi.instruments.append(inst)

        return midi

    def words_to_events(self, words: [str]) -> [Event]:
        events = []
        for word in words:
            if '_' not in word:
                print(f"{word} could not be converted to an event.")
                continue
            event_name, event_value = word.split('_')
            events.append(Event(event_name, None, event_value, None))
        return events
