import miditoolkit
import numpy as np

from mgt.datamanagers.data_manager import Dictionary
from mgt.datamanagers.remi import chord_recognition
from mgt.datamanagers.remi.constants import DRUM_INSTRUMENT, DEFAULT_RESOLUTION, DEFAULT_FRACTION, \
    DEFAULT_VELOCITY_BINS, DEFAULT_DURATION_BINS, DEFAULT_TEMPO_INTERVALS
from mgt.datamanagers.remi.event import Event
from mgt.datamanagers.remi.item import Item


class DataExtractor(object):

    def __init__(
            self,
            dictionary: Dictionary,
            map_tracks_to_instruments: {},
            use_chords: bool,
            instrument_mapping: {}
    ):
        self.dictionary = dictionary
        self.map_tracks_to_instruments = map_tracks_to_instruments
        self.use_chords = use_chords
        self.instrument_mapping = instrument_mapping

    def extract_data(self, path: str, transposition_steps: int):
        print(f"Extracting data for {path}")
        words = self.extract_words(path, transposition_steps)
        return self.words_to_data(words)

    def words_to_data(self, words):
        return list(map(lambda x: self.dictionary.word_to_data(x), words))

    def events_to_words(self, events):
        return list(map(lambda x: '{}_{}'.format(x.name, x.value), events))

    def extract_words(self, path: str, transposition_steps: int):
        events = self.extract_events(path, transposition_steps)
        return self.events_to_words(events)

    def extract_events(self, input_path: str, transposition_steps: int):
        if transposition_steps != 0:
            print("Transposing {} steps.".format(transposition_steps))

        note_items, tempo_items = self.read_items(
            input_path,
            transposition_steps
        )

        note_items = self.quantize_items(note_items)
        max_time = note_items[-1].end
        if self.use_chords:
            chord_items = self.extract_chords(note_items)
            items = chord_items + tempo_items + note_items
        else:
            items = tempo_items + note_items
        groups = self.group_items(items, max_time)
        events = self.item2event(groups)
        return events

    def read_items(self, file_path: str, transposition_steps: int):
        midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
        # note
        notes = []
        note_items = []
        for index, instrument in enumerate(midi_obj.instruments):
            if index in self.map_tracks_to_instruments:
                program = self.map_tracks_to_instruments.get(index)
            else:
                program = DRUM_INSTRUMENT if instrument.is_drum else instrument.program
                if program in self.instrument_mapping:
                    program = self.instrument_mapping[program]
                    if program is None:
                        continue
            for note in instrument.notes:
                notes.append({'note': note, 'instrument': program})

        notes.sort(key=lambda x: (x['note'].start, x['instrument'], x['note'].pitch))
        for note in notes:
            adjusted_pitch = note['note'].pitch
            if note['instrument'] != DRUM_INSTRUMENT:
                adjusted_pitch += transposition_steps

                # To prevent invalid pitches
                if adjusted_pitch < 0:
                    adjusted_pitch += 12
                if adjusted_pitch > 127:
                    adjusted_pitch -= 12

            note_items.append(Item(
                name='Note',
                start=note['note'].start,
                end=note['note'].end,
                velocity=note['note'].velocity,
                instrument=note['instrument'],
                pitch=adjusted_pitch))
        note_items.sort(key=lambda x: x.start)
        # tempo
        tempo_items = []
        for tempo in midi_obj.tempo_changes:
            tempo_items.append(Item(
                name='Tempo',
                start=tempo.time,
                end=None,
                velocity=None,
                pitch=int(tempo.tempo)))
        tempo_items.sort(key=lambda x: x.start)
        # expand to all beat
        max_tick = tempo_items[-1].start
        existing_ticks = {item.start: item.pitch for item in tempo_items}
        wanted_ticks = np.arange(0, max_tick + 1, DEFAULT_RESOLUTION)
        output = []
        for tick in wanted_ticks:
            if tick in existing_ticks:
                output.append(Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=existing_ticks[tick]))
            else:
                output.append(Item(
                    name='Tempo',
                    start=tick,
                    end=None,
                    velocity=None,
                    pitch=output[-1].pitch))
        tempo_items = output
        return note_items, tempo_items

    def quantize_items(self, items: [Item], ticks=120):
        # grid
        grids = np.arange(0, items[-1].start, ticks, dtype=int)
        # process
        for item in items:
            index = np.argmin(abs(grids - item.start))
            shift = grids[index] - item.start
            item.start += shift
            item.end += shift
        return items

    def extract_chords(self, items: [Item]):
        method = chord_recognition.MIDIChord()

        items_without_drums = list(filter(lambda item: item.instrument != DRUM_INSTRUMENT, items))

        chords = method.extract(notes=items_without_drums)
        output = []
        for chord in chords:
            output.append(Item(
                name='Chord',
                start=chord[0],
                end=chord[1],
                velocity=None,
                pitch=chord[2].split('/')[0]))
        return output

    def group_items(self, items: [Item], max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
        items.sort(key=lambda x: x.start)
        downbeats = np.arange(0, max_time + ticks_per_bar, ticks_per_bar)
        groups = []
        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            insiders = []
            for item in items:
                if (item.start >= db1) and (item.start < db2):
                    insiders.append(item)
            overall = [db1] + insiders + [db2]
            groups.append(overall)
        return groups

    def item2event(self, groups):
        events = []
        n_downbeat = 0
        for i in range(len(groups)):
            if 'Note' not in [item.name for item in groups[i][1:-1]]:
                continue
            bar_st, bar_et = groups[i][0], groups[i][-1]
            n_downbeat += 1
            events.append(Event(
                name='Bar',
                time=None,
                value=None,
                text='{}'.format(n_downbeat)))
            for item in groups[i][1:-1]:
                # position
                flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
                index = np.argmin(abs(flags - item.start))
                events.append(Event(
                    name='Position',
                    time=item.start,
                    value='{}/{}'.format(index + 1, DEFAULT_FRACTION),
                    text='{}'.format(item.start)))
                if item.name == 'Note':
                    # instrument
                    events.append(Event(
                        name='Instrument',
                        time=item.start,
                        value=item.instrument,
                        text='{}'.format(item.instrument)))

                    # velocity
                    velocity_index = np.searchsorted(
                        DEFAULT_VELOCITY_BINS,
                        item.velocity,
                        side='right') - 1
                    events.append(Event(
                        name='Note Velocity',
                        time=item.start,
                        value=velocity_index,
                        text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                    # pitch
                    events.append(Event(
                        name='Note On',
                        time=item.start,
                        value=item.pitch,
                        text='{}'.format(item.pitch)))
                    # duration
                    duration = item.end - item.start
                    index = np.argmin(abs(DEFAULT_DURATION_BINS - duration))
                    events.append(Event(
                        name='Note Duration',
                        time=item.start,
                        value=index,
                        text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
                elif item.name == 'Chord':
                    events.append(Event(
                        name='Chord',
                        time=item.start,
                        value=item.pitch,
                        text='{}'.format(item.pitch)))
                elif item.name == 'Tempo':
                    tempo = item.pitch
                    if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                        tempo_style = Event('Tempo Class', item.start, 'slow', None)
                        tempo_value = Event('Tempo Value', item.start,
                                            tempo - DEFAULT_TEMPO_INTERVALS[0].start, None)
                    elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                        tempo_style = Event('Tempo Class', item.start, 'mid', None)
                        tempo_value = Event('Tempo Value', item.start,
                                            tempo - DEFAULT_TEMPO_INTERVALS[1].start, None)
                    elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                        tempo_style = Event('Tempo Class', item.start, 'fast', None)
                        tempo_value = Event('Tempo Value', item.start,
                                            tempo - DEFAULT_TEMPO_INTERVALS[2].start, None)
                    elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                        tempo_style = Event('Tempo Class', item.start, 'slow', None)
                        tempo_value = Event('Tempo Value', item.start, 0, None)
                    elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                        tempo_style = Event('Tempo Class', item.start, 'fast', None)
                        tempo_value = Event('Tempo Value', item.start, 59, None)
                    events.append(tempo_style)
                    events.append(tempo_value)
        return events
