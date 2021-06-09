from pretty_midi import PrettyMIDI

from mgt.datamanagers.time_shift.time_util import TimeUtil


class Event(object):
    def __init__(self, event_type, start, data):
        self.event_type = event_type
        self.start = start
        self.data = data

    def __repr__(self):
        return 'Event(type={}, start={}, data={})'.format(
            self.event_type, self.start, self.data)


def event_type_sorting(event_type):
    if event_type == "start-track":
        return 0
    elif event_type == "note":
        return 1
    elif event_type == "time-shift":
        return 2


def program_sort(event):
    program = 0
    if 'program' in event.data:
        return program


def velocity_sort(event):
    velocity = 0
    if 'velocity' in event.data:
        return velocity


def pitch_sort(event):
    pitch = 0
    if 'pitch' in event.data:
        return pitch


class EventExtractor(object):

    def __init__(self, bins_per_measure=32):
        self.bins_per_measure = bins_per_measure

    def extract_events(self, midi_data: PrettyMIDI):
        time_bins = TimeUtil.divide_midi_into_bins(midi_data, self.bins_per_measure)
        all_events = []
        all_events.extend(self.extract_notes(midi_data, time_bins))
        all_events.sort(key=lambda x: x.start)
        all_events.insert(0, Event(event_type="start-track", start=0, data={}))
        all_events.extend(self.create_time_shift_events(all_events))
        all_events.sort(key=lambda x: (
            x.start,
            event_type_sorting(x.event_type),
            program_sort(x),
            velocity_sort(x)
        ))
        all_events.append(Event(event_type="end-track", start=all_events[len(all_events) - 1].start, data={}))
        return all_events

    @classmethod
    def extract_notes(cls, midi_data: PrettyMIDI, time_bins):
        events = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                program = 128 if instrument.is_drum else instrument.program
                start = TimeUtil.time_to_time_bin(time_bins, note.start)
                end = TimeUtil.time_to_time_bin(time_bins, note.end)
                events.append(Event(
                    event_type="note",
                    start=TimeUtil.time_to_time_bin(time_bins, note.start),
                    data={
                        "program": program,
                        "velocity": int(note.velocity / 4),
                        "duration": max(end - start, 1),
                        "pitch": note.pitch
                    }
                ))

        events.sort(key=lambda x: x.start)
        return events

    def create_bar_events(self, time_bins):
        bar_events = []
        for i in range(0, len(time_bins), self.bins_per_measure):
            bar_events.append(Event(event_type="bar", start=i, data={}))
        return bar_events

    def create_time_shift_events(self, events: [Event]):
        time_shift_events = []
        for i in range(1, len(events)):
            prev_event = events[i - 1]
            event = events[i]
            time_shift = event.start - prev_event.start
            if time_shift > 0:
                time_shift_events.append(
                    Event(
                        event_type="time-shift",
                        start=prev_event.start,
                        data={"duration": time_shift}
                    )
                )
        return time_shift_events
