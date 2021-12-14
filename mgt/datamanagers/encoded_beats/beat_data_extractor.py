from pretty_midi import PrettyMIDI, np, pretty_midi

from mgt.datamanagers.encoded_beats.encoded_beats_constants import POSSIBLE_MIDI_PITCHES


def get_closest(number, target1, target2):
    return target2 if abs(target1 - number) > abs(target2 - number) else target1


def append_to_matrix(matrix, program, pitch, activated_sub_beats):
    for i in range(activated_sub_beats[0], activated_sub_beats[1] + 1):
        if i == activated_sub_beats[1]:
            matrix[i][program][pitch] = 2
        else:
            matrix[i][program][pitch] = 1


defaults = {
    'instruments': {
        27,  # Electric guitar
        70,  # Bassoon
        33,  # Electric bass
        128  # Drums
    },
    'beat_resolution': 4
}


class BeatDataExtractor(object):

    def __init__(
            self,
            instruments: [int] = defaults['instruments'],       # Which instrument should be used per midi track
            beat_resolution: int = defaults['beat_resolution']  # In how many pieces should the beats be divided
    ):
        self.instruments = instruments
        self.beat_resolution = beat_resolution

    def extract_beats(self, midi_data: PrettyMIDI):
        beats = midi_data.get_beats().tolist()
        last_tempo = midi_data.get_tempo_changes()[-1][0]
        beats.append(beats[-1] + (60 / last_tempo))
        subdivided_beats = self.subdivide_beats(beats)
        sub_beat_matrices = self.get_sub_beat_matrices(subdivided_beats, midi_data)
        beat_matrices = []
        for i in range(0, len(subdivided_beats) - self.beat_resolution, self.beat_resolution):
            beat_matrices.append(sub_beat_matrices[i: i + 4])
        beat_matrices = np.array(beat_matrices)
        beat_matrices = beat_matrices.reshape((len(beat_matrices), len(self.instruments) * self.beat_resolution * POSSIBLE_MIDI_PITCHES))
        print(f"Created a matrix with shape {beat_matrices.shape}")
        return np.array(beat_matrices)

    def restore_midi(self, model_output) -> pretty_midi.PrettyMIDI:
        reshaped = model_output.reshape((len(model_output), len(self.instruments), self.beat_resolution, POSSIBLE_MIDI_PITCHES))

        notes = []

        for beat_index, sub_beats in enumerate(reshaped):
            for sub_beat_index, tracks in enumerate(sub_beats):
                for track_index, pitches in enumerate(tracks):
                    for pitch, note_type in enumerate(pitches):
                        if note_type == 1:
                            notes.append({'track': track_index, 'pitch': pitch,
                                          'time': beat_index * 4 + sub_beat_index, 'type': 'on'})
                        elif note_type == 2:
                            notes.append({'track': track_index, 'pitch': pitch,
                                          'time': beat_index * 4 + sub_beat_index, 'type': 'off'})

        sorted_notes = sorted(notes, key=lambda x: (x['instrument'], x['time']))

        midi = pretty_midi.PrettyMIDI()

        for program in self.instruments:
            instrument = pretty_midi.Instrument(program=program)
            midi.instruments.append(instrument)

        prev_note = None
        current_midi_note = None
        for note in sorted_notes:
            instrument = midi.instruments[note['track']]
            time = note['time']
            if prev_note is not None \
                    and note['track'] == prev_note['track'] \
                    and note['pitch'] == prev_note['pitch'] \
                    and time == prev_note['time'] + 1:
                current_midi_note['end'] += 1
            else:
                if current_midi_note is not None:
                    midi_note = pretty_midi.Note(
                        velocity=64,
                        pitch=current_midi_note['pitch'],
                        start=current_midi_note['start'] * (0.5 / 4),
                        end=current_midi_note['end'] * (0.5 / 4)
                    )
                    instrument.notes.append(midi_note)

                current_midi_note = {
                    'track': note['track'],
                    'pitch': note['pitch'],
                    'start': note['time'],
                    'end': note['time'] + 1
                }
            prev_note = note

        return midi

    def subdivide_beats(self, beats):
        subdivided = []
        prev_beat = 0
        for beat in beats[1:]:
            step_size = (beat - prev_beat) / self.beat_resolution
            prev_step = prev_beat
            for _ in range(self.beat_resolution):
                subdivided.append(prev_step)
                prev_step += step_size
            prev_beat = beat

        subdivided.append(prev_beat)

        return subdivided

    def get_sub_beat_matrices(self, subdivided_beats, midi_data):
        matrix = []
        for x in range(len(subdivided_beats) - 1):  # Sub beats
            matrix.append([])
            for y in range(4):  # Instruments
                matrix[x].append([])
                for z in range(128):  # Pitches
                    matrix[x][y].append(0)

        for index, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                activated_sub_beats = self.get_activated_sub_beats(note.start, note.end, subdivided_beats)
                append_to_matrix(matrix, index, note.pitch, activated_sub_beats)

        return matrix

    def get_activated_sub_beats(self, note_start, note_end, subdivided_beats):
        sub_beat_start = self.find_nearest_matching_sub_beat(note_start, subdivided_beats)
        sub_beat_end = self.find_nearest_matching_sub_beat(note_end, subdivided_beats) - 1
        return sub_beat_start, sub_beat_end

    def find_nearest_matching_sub_beat(self, time, subdivided_beats):
        prev_sub_beat = 0
        for index, sub_beat in enumerate(subdivided_beats[1:]):
            if prev_sub_beat <= time <= sub_beat:
                nearest_match = get_closest(time, prev_sub_beat, sub_beat)
                return index if nearest_match == prev_sub_beat else index + 1
            prev_sub_beat = sub_beat
