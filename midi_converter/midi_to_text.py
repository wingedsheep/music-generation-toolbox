import pretty_midi


# def get_tempo_items(midi_data):
#     tempo_changes = midi_data.get_tempo_changes()
#
#
#
#     tempo_bins = [[10, 15, 0]]
#     tempo_changes_binned = bin_tempo_changes(tempo_changes, 20




def midi_to_text(midi_path):
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # tempo_items = get_tempo_items(midi_data)
    print(midi_data.get_tempo_changes())
    print(midi_data.get_beats())



# midi_to_text("/Users/vincentbons/Documents/AI Song contest/midi/voor muziek/The Weeknd - Blinding Lights (full).midi")
midi_to_text("/Users/vincentbons/Documents/AI Song contest/midi/raw/NewBorn.midi")
