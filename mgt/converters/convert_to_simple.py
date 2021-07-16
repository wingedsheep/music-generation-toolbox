import pretty_midi

from mgt.converters.chord_name_to_pitches import chord_name_to_pitches
from mgt.datamanagers.midi_wrapper import PrettyMidiWrapper
from mgt.datamanagers.remi.util import extract_events
from stats_extraction import extract_stats


def get_chords(input_path):
    chord_events = extract_events(input_path, only_chords=True)
    return chord_events


def get_bass_notes(midi):
    return midi.instruments[1].notes


def add_bass(midi, new_midi):
    bass_notes = get_bass_notes(midi)
    instrument = pretty_midi.Instrument(program=35)
    new_midi.instruments.append(instrument)
    instrument.notes.extend(bass_notes)


def get_vocal_melody_notes(midi):
    return midi.instruments[6].notes


def add_vocal_melody(midi, new_midi):
    vocal_melody_notes = get_vocal_melody_notes(midi)
    instrument = pretty_midi.Instrument(program=67)
    new_midi.instruments.append(instrument)
    instrument.notes.extend(vocal_melody_notes)


def convert_midi_to_simple(input_path, output_path):
    """
    Convert midi to a midi split into tracks by category
    * Vocal melody
    * Instrumental melody
    * Bass
    * Chords
    * Drums
    """
    midi = pretty_midi.PrettyMIDI(input_path, resolution=480)

    stats = extract_stats(midi)

    new_midi = pretty_midi.PrettyMIDI()

    add_chords(input_path, midi, new_midi)
    add_drums(midi, new_midi)
    add_bass(midi, new_midi)
    add_vocal_melody(midi, new_midi)

    PrettyMidiWrapper(new_midi).save(output_path)


def add_chords(input_path, midi, new_midi):
    chords = get_chords(input_path)
    instrument = pretty_midi.Instrument(program=51, name="Synth Strings 3")  # Violin for chords
    new_midi.instruments.append(instrument)
    for i in range(1, len(chords) - 1):
        prev_chord = chords[i - 1]
        current_chord = chords[i]
        pitches = chord_name_to_pitches(prev_chord.value)
        for pitch in pitches:
            note = pretty_midi.Note(velocity=60, start=prev_chord.time / 480 / 2, end=current_chord.time / 480 / 2, pitch=pitch)
            instrument.notes.append(note)
    last_chord = chords[len(chords) - 1]
    pitches = chord_name_to_pitches(last_chord.value)
    for pitch in pitches:
        note = pretty_midi.Note(velocity=60, start=last_chord.time / 480 / 2, end=midi.get_end_time(), pitch=pitch)
        instrument.notes.append(note)


def get_drums(midi: pretty_midi.PrettyMIDI):
    drum_notes = []
    instrument: pretty_midi.Instrument
    for instrument in midi.instruments:
        if instrument.is_drum:
            for note in instrument.notes:
                drum_notes.append(note)
    return drum_notes


def add_drums(midi, new_midi):
    drum_notes = get_drums(midi)
    instrument = pretty_midi.Instrument(program=1, is_drum=True)
    new_midi.instruments.append(instrument)
    instrument.notes.extend(drum_notes)


convert_midi_to_simple(
    "/Users/vincentbons/Documents/Music toolbox/test separation/input/Miley Cyrus - Wrecking ball.midi",
    "test.midi")
