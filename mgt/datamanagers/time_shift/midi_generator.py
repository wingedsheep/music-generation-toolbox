import pretty_midi

from mgt.datamanagers.time_shift.event_extractor import Event


class MidiGenerator(object):

    @classmethod
    def events_to_midi(cls, events: [Event], starting_tempo=120) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI(resolution=480)
        events_per_instrument = cls.get_events_per_instrument(events)
        time_per_tick = (60 / starting_tempo / 32) * 4

        for original_program in events_per_instrument:
            is_drum = original_program == 128
            program = 1 if original_program == 128 else original_program
            instrument = pretty_midi.Instrument(program=program)
            instrument.is_drum = is_drum
            for event in events_per_instrument[original_program]:
                start_time = event.start * time_per_tick
                duration = event.data["duration"] * time_per_tick
                end_time = start_time + duration
                note = pretty_midi.Note(
                    velocity=event.data["velocity"] * 4,
                    pitch=event.data["pitch"],
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
            midi.instruments.append(instrument)

        return midi

    @staticmethod
    def get_events_per_instrument(events: [Event]):
        events_per_instrument = {}
        for event in events:
            if event.event_type != 'note':
                continue

            program = event.data["program"]
            if program not in events_per_instrument:
                events_per_instrument[program] = []
            events_per_instrument[program].append(event)
        return events_per_instrument
