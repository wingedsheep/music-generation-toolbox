def chord_name_to_pitches(chord):
    base_chord = []

    if ':aug' in chord:
        base_chord = [48, 52, 56]
    elif ':dim' in chord:
        base_chord = [48, 51, 54]
    elif ':dom' in chord:
        base_chord = [46, 50, 53, 56]
    elif ':maj' in chord:
        base_chord = [48, 52, 55]
    elif ':min' in chord:
        base_chord = [48, 51, 55]

    if 'C:' in chord:
        return base_chord
    elif 'C#:' in chord:
        return [x + 1 for x in base_chord]
    elif 'D:' in chord:
        return [x + 2 for x in base_chord]
    elif 'D#:' in chord:
        return [x + 3 for x in base_chord]
    elif 'E:' in chord:
        return [x + 4 for x in base_chord]
    elif 'F:' in chord:
        return [x + 5 for x in base_chord]
    elif 'F#:' in chord:
        return [x + 6 for x in base_chord]
    elif 'G:' in chord:
        return [x + 7 for x in base_chord]
    elif 'G#:' in chord:
        return [x + 8 for x in base_chord]
    elif 'A:' in chord:
        return [x + 9 for x in base_chord]
    elif 'A#:' in chord:
        return [x + 10 for x in base_chord]
    elif 'B:' in chord:
        return [x + 11 for x in base_chord]

    print(f"{chord} could not be mapped to pitches")
