from pretty_midi import PrettyMIDI, np, pretty_midi

from mgt.datamanagers.encoded_beats.encoded_beats_constants import POSSIBLE_MIDI_PITCHES


def get_closest(number, target1, target2):
    return target2 if abs(target1 - number) > abs(target2 - number) else target1


def append_to_matrix(matrix, program, pitch, activated_sub_beats):
    """
    1 is the start of a new note
    """
    matrix[activated_sub_beats[0]][program][pitch] = 1
    if activated_sub_beats[1] > activated_sub_beats[0]:
        for sub_beat in range(activated_sub_beats[0] + 1, activated_sub_beats[1] + 1):
            matrix[sub_beat][program][pitch] = 1


defaults = {
    'tracks': [
        27,  # Electric guitar
        70,  # Bassoon
        33,  # Electric bass
        128  # Drums
    ],
    'beat_resolution': 4
}


class BeatDataExtractor(object):

    def __init__(
            self,
            tracks: [int] = defaults['tracks'],  # Which instrument should be used per midi track
            beat_resolution: int = defaults['beat_resolution']  # In how many pieces should the beats be divided
    ):
        self.tracks = tracks
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
        beat_matrices = beat_matrices.reshape(
            (len(beat_matrices), 128 * self.beat_resolution * POSSIBLE_MIDI_PITCHES))
        print(f"Created a matrix with shape {beat_matrices.shape}")
        return np.array(beat_matrices)

    def restore_midi(self, model_output) -> pretty_midi.PrettyMIDI:
        output = np.array(model_output)
        reshaped = output.reshape((len(output), self.beat_resolution, 128, POSSIBLE_MIDI_PITCHES))

        notes = []

        for beat_index, sub_beats in enumerate(reshaped):
            for sub_beat_index, instruments in enumerate(sub_beats):
                for track_index, pitches in enumerate(instruments):
                    for pitch, activation_type in enumerate(pitches):
                        if activation_type > 0:
                            notes.append({'instrument': track_index, 'pitch': pitch,
                                          'time': beat_index * 4 + sub_beat_index, 'activation_type': activation_type})

        sorted_notes = sorted(notes, key=lambda x: (x['track'], x['pitch'], x['time']))

        midi = pretty_midi.PrettyMIDI()

        for program in self.tracks:
            instrument = pretty_midi.Instrument(program=program)
            midi.instruments.append(instrument)

        sub_beat_duration = (0.5 / 4)  # TODO not hardcoded
        current_note = None
        for note in sorted_notes:
            instrument = midi.instruments[note['track']]
            time = note['time']
            pitch = note['pitch']
            activation_type = note['activation_type']

            if current_note is not None and \
                    activation_type != 1 and \
                    current_note['instrument'] == instrument and \
                    current_note['start'] + current_note['duration'] == time and \
                    current_note['pitch'] == pitch:
                current_note['duration'] = current_note['duration'] +1
            else:
                if current_note is not None:
                    midi_note = pretty_midi.Note(
                        velocity=64,
                        pitch=current_note['pitch'],
                        start=current_note['start'] * sub_beat_duration,
                        end=(current_note['start'] + current_note['duration']) * sub_beat_duration
                    )
                    current_note['instrument'].notes.append(midi_note)
                current_note = {
                    'instrument': instrument,
                    'start': time,
                    'duration': 1,
                    'pitch': pitch
                }

        if current_note is not None:
            midi_note = pretty_midi.Note(
                velocity=64,
                pitch=current_note['pitch'],
                start=current_note['start'] * sub_beat_duration,
                end=(current_note['start'] + current_note['duration']) * sub_beat_duration
            )
            current_note['instrument'].notes.append(midi_note)

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
            for y in range(128):  # Instruments
                matrix[x].append([])
                for z in range(128):  # Pitches
                    matrix[x][y].append(0)

        for index, instrument in enumerate(midi_data.instruments):
            for note in instrument.notes:
                activated_sub_beats = self.get_activated_sub_beats(note.start, note.end, subdivided_beats)
                append_to_matrix(matrix, index, note.pitch, activated_sub_beats)

        return matrix

    def get_activated_sub_beats(self, note_start, note_end, subdivided_beats):
        # TODO replace with finding subbeats directly based on time
        sub_beat_start = self.find_nearest_matching_sub_beat(note_start, subdivided_beats)
        sub_beat_end = self.find_nearest_matching_sub_beat(note_end, subdivided_beats)
        return sub_beat_start, sub_beat_end

    def find_nearest_matching_sub_beat(self, time, subdivided_beats):
        prev_sub_beat = 0
        index = 0
        for index, sub_beat in enumerate(subdivided_beats[1:]):
            if prev_sub_beat <= time <= sub_beat:
                nearest_match = get_closest(time, prev_sub_beat, sub_beat)
                return index if nearest_match == prev_sub_beat else index + 1
            prev_sub_beat = sub_beat
        return index
