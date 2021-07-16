import pretty_midi
import numpy as np


def average_pitch(instrument: pretty_midi.Instrument):
    pitches = [note.pitch for note in instrument.notes]
    return np.average(pitches)


def average_number_of_simultaneous_notes(instrument: pretty_midi.Instrument):
    bins = {}
    note: pretty_midi.Note
    for note in instrument.notes:
        if note.start not in bins:
            bins[note.start] = 1
        else:
            bins[note.start] += 1
    return np.average(list(bins.values()))


def note_repetition_value(instrument: pretty_midi.Instrument):
    """
    For every time shift in the song, return a value indicating if the notes have changed.
    Going from a C to a G returns a score of 0.
    Going from a chord C, E, G to a chord C, D#, G returns a score of 2/3 since 2 out of 3 of the notes remain the same.
    The average is taken over all these scores
    :param instrument:
    :return:
    """
    note: pretty_midi.Note
    bins = []
    current_start = -1
    for note in instrument.notes:
        if note.start != current_start:
            bins.append([note.pitch])
        else:
            bins[len(bins) - 1].append(note.pitch)
        current_start = note.start

    repetition_values = []
    for i in range(1, len(bins) - 1):
        previous_bin = bins[i - 1]
        current_bin = bins[i]
        difference = np.setdiff1d(previous_bin, current_bin)
        all_notes = list(set(previous_bin) | set(current_bin))
        score = len(difference) / len(all_notes)
        repetition_values.append(score)

    return np.average(repetition_values)


def average_duration(instrument):
    durations = [note.duration for note in instrument.notes]
    return np.average(durations)


def extract_stats(midi):
    average_pitches = []
    instruments = []
    average_simultaneous_notes = []
    repetition_values = []
    average_durations = []
    for instrument in midi.instruments:
        average_pitches.append(average_pitch(instrument))
        instruments.append(128 if instrument.is_drum else instrument.program)
        average_simultaneous_notes.append(average_number_of_simultaneous_notes(instrument))
        repetition_values.append(note_repetition_value(instrument))
        average_durations.append(average_duration(instrument))

    return {
        'average_pitches': average_pitches,
        'instruments': instruments,
        'average_simultaneous': average_simultaneous_notes,
        'repetition_values': repetition_values,
        'average_durations': average_durations
    }
