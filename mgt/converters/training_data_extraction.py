from pretty_midi import pretty_midi
import numpy as np

from mgt.converters.stats_extraction import extract_stats

midi = pretty_midi.PrettyMIDI("/Users/vincentbons/Documents/Music toolbox/test separation/input/Adele - Rolling in the deep.midi", resolution=480)
stats = extract_stats(midi)

input = []
labels = []

number_of_tracks = len(stats['average_pitches'])


def one_hot_encode(value, number_of_labels):
    one_hot = np.zeros(number_of_labels)
    one_hot[value] = 1
    return list(one_hot)


for i in range(number_of_tracks):
    input_data = [stats['average_pitches'][i]]
    input_data.extend(one_hot_encode(stats['instruments'][i], 129))
    input_data.append(stats['average_simultaneous'][i])
    input_data.append(stats['repetition_values'][i])
    input_data.append(stats['average_durations'][i])
    input.append(input_data)


# 0 = vocals
# 1 = instrumental_melody
# 2 = bass
# 3 = drums
# 4 = accompaniment
categories = 5
labels = [2, 0, 4, 4, 4, 3]

print(input)
print(labels)
