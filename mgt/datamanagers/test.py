import mido


def midi_to_tokens(midi_file, bar_length):
    # Open the MIDI file
    midi = mido.MidiFile(midi_file)

    # Set the initial tempo and time signature
    tempo = midi.ticks_per_beat
    time_signature = (4, 4)  # Default to 4/4 time

    # Initialize the tokens list and other variables
    tokens = []
    current_beat = 0
    current_bar = 0
    current_instrument = 0
    timing = 0  # Initialize the timing variable

    # Iterate through the MIDI events
    for track in midi.tracks:
        for message in track:
            # Update the current instrument if necessary
            if message.type == "program_change":
                current_instrument = message.program

            # Update the time signature if necessary
            if message.type == "time_signature":
                time_signature = message.numerator, message.denominator

            # Update the tempo if necessary
            if message.type == "set_tempo":
                tempo = message.tempo

            # If this is a note event, add the appropriate tokens
            if message.type == "note_on" or message.type == "note_off":
                # Calculate the length of the note in beats
                if timing == 0:
                    # If this is the first note, use the tempo and time signature to calculate the timing
                    timing = tempo / time_signature[1]  # timing is the number of ticks per beat
                note_length = message.time / timing
                note_beats = int(note_length)
                note_fraction = note_length - note_beats

                # Add tokens for the start and end of the note
                if message.type == "note_on":
                    tokens.append(f"NOTE_START_{message.note}")
                else:
                    tokens.append(f"NOTE_END_{message.note}")

                # Add a token for the volume of the note
                tokens.append(f"VOLUME_{message.velocity}")

                # Add a token for the current instrument
                tokens.append(f"INSTRUMENT_{current_instrument}")

                # Add tokens for each beat of the note
                for i in range(note_beats):
                    # If this is the start of a new bar, add a bar token
                    if current_beat % bar_length == 0:
                        tokens.append(f"BAR_{current_bar}")
                        current_bar += 1

                    # Add a beat token
                    tokens.append(f"BEAT_{current_beat}")
                    current_beat += 1

                # If there's a fractional part of the note left, add a token for that as well
                if note_fraction > 0:
                    tokens.append(f"BEAT_{current_beat}_FRACTION_{note_fraction:.2f}")
                    current_beat += note_fraction

    # Add tokens for the end of the song
    tokens.append(f"END_SONG_{current_beat}")

    return tokens


tokens = midi_to_tokens("/Users/vincent/Git/music-generation-toolbox/data/pop/001.mid", 480)

print(len(tokens))
