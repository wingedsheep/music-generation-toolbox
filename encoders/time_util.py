from pretty_midi import PrettyMIDI


class TimeUtil(object):

    @staticmethod
    def divide_midi_into_bins(midi_data: PrettyMIDI, bins_per_bar):
        end_time = midi_data.get_end_time()
        tempo_changes = midi_data.get_tempo_changes()
        current_tempo_index = 0
        tempo = tempo_changes[1][current_tempo_index]
        next_tempo_time = tempo_changes[0][current_tempo_index + 1] \
            if tempo_changes[0].size > current_tempo_index + 1 else None
        seconds_per_bin = 60 / tempo * 4 / bins_per_bar
        current_time = 0
        time_bins = []
        while current_time < end_time:
            time_bins.append(current_time)
            current_time += seconds_per_bin
            current_time = round(current_time, 2)
            if next_tempo_time is not None and current_time >= next_tempo_time:
                current_tempo_index += 1
                tempo = tempo_changes[1][current_tempo_index]
                next_tempo_time = tempo_changes[0][current_tempo_index + 1] \
                    if tempo_changes[0].size > current_tempo_index + 1 else None
                seconds_per_bin = 60 / tempo * 4 / bins_per_bar
        return time_bins

    @staticmethod
    def time_to_time_bin(time_bins, time):
        for index in range(len(time_bins) - 1):
            bin_start_time = time_bins[index]
            bin_end_time = time_bins[index + 1]
            if bin_start_time <= time < bin_end_time:
                return index
        return len(time_bins) - 1

    @staticmethod
    def tempo_to_bin(tempo, number_of_bins=40, items_per_bin=5, starting_tempo=10):
        tempo_bin = tempo - starting_tempo / items_per_bin
        if tempo_bin >= number_of_bins:
            tempo_bin = number_of_bins - 1
        return tempo_bin
