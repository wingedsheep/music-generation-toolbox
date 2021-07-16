import numpy as np
import miditoolkit
import copy

# parameters for input
from mgt.datamanagers.remi import chord_recognition
from mgt.datamanagers.time_shift.event_extractor import Event

DRUM_INSTRUMENT = 128

DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480


# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch, instrument=None):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
        self.instrument = instrument

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={}, instrument={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch, self.instrument)


# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path, transposition_steps=0, map_tracks_to_instruments=None):
    if map_tracks_to_instruments is None:
        map_tracks_to_instruments = {}

    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    notes = []
    note_items = []
    for index, instrument in enumerate(midi_obj.instruments):
        if index in map_tracks_to_instruments:
            program = map_tracks_to_instruments.get(index)
        else:
            program = DRUM_INSTRUMENT if instrument.is_drum else instrument.program
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


# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items


# extract chord
def extract_chords(items: [Item]):
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


def extract_events(input_path,
                   transposition_steps=0,
                   map_tracks_to_instruments=None,
                   use_chords=True,
                   only_chords=False) -> [Event]:

    if map_tracks_to_instruments is None:
        map_tracks_to_instruments = {}

    if transposition_steps != 0:
        print("Transposing {} steps.".format(transposition_steps))

    note_items, tempo_items = read_items(
        input_path,
        transposition_steps=transposition_steps,
        map_tracks_to_instruments=map_tracks_to_instruments)

    note_items = quantize_items(note_items)
    max_time = note_items[-1].end
    if use_chords:
        chord_items = extract_chords(note_items)
        items = chord_items + tempo_items + note_items
    else:
        items = tempo_items + note_items
    groups = group_items(items, max_time)
    events = item2event(groups)

    if only_chords:
        events = list(filter(lambda x: x.name == 'Chord' and x.value != 'N:N', events))

    return events


# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION * 4):
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


# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)


# item to event
def item2event(groups):
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


#############################################################################################
# WRITE MIDI
#############################################################################################
def write_midi(words, word2event, output_path, prompt_path=None, bars_in_prompt=4):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events) - 3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
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
        elif events[i].name == 'Position' and events[i + 1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i + 1].value])
        elif events[i].name == 'Position' and \
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
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        last_time = DEFAULT_RESOLUTION * 4 * bars_in_prompt

        new_midi = miditoolkit.midi.parser.MidiFile()
        new_midi.ticks_per_beat = DEFAULT_RESOLUTION

        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < last_time:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        new_midi.tempo_changes = temp_tempos

        existing_notes = {}
        for instrument in midi.instruments:
            program = DRUM_INSTRUMENT if instrument.is_drum else instrument.program
            for note in instrument.notes:
                if note.end <= last_time:
                    existing_notes.setdefault(program, []).append(note)

        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                new_midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0] + last_time))

        for instrument, instrument_notes in notes.items():
            program = 0 if instrument == DRUM_INSTRUMENT else instrument
            is_drum = True if instrument == DRUM_INSTRUMENT else False
            inst = miditoolkit.midi.containers.Instrument(program, is_drum=is_drum)

            if instrument in existing_notes:
                inst.notes.extend(existing_notes[instrument])

            # note shift and add
            for note in instrument_notes:
                note.start += last_time
                note.end += last_time
                inst.notes.append(note)

            new_midi.instruments.append(inst)

        # write
        new_midi.dump(output_path)
    else:
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

        # write
        midi.dump(output_path)
    print(f"Written midi to {output_path}")


def extract_data(path,
                 dictionary,
                 transposition_steps=0,
                 map_tracks_to_instruments=None,
                 use_chords=True,
                 only_chords=False):
    if map_tracks_to_instruments is None:
        map_tracks_to_instruments = {}
    print(f"Extracting data for {path}")
    events = extract_events(path,
                            transposition_steps=transposition_steps,
                            map_tracks_to_instruments=map_tracks_to_instruments,
                            use_chords=use_chords,
                            only_chords=only_chords)
    words = list(map(lambda x: '{}_{}'.format(x.name, x.value), events))
    data = list(map(lambda x: dictionary.word_to_data(x), words))
    return data


def words_to_events(words):
    events = []
    for word in words:
        if '_' not in word:
            print(f"{word} could not be converted to an event.")
            continue
        event_name, event_value = word.split('_')
        events.append(Event(event_name, None, event_value, None))
    return events


def to_midi(data, dictionary):
    words = list(map(lambda x: dictionary.data_to_word(x), data))
    events: [Event] = words_to_events(words)

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
