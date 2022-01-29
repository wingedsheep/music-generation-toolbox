from enum import IntEnum

from mgt.datamanagers.remi.event import Event


class RemiEventType(IntEnum):
    BAR = 0
    TEMPO = 1
    NOTE = 2


class RemiItem(object):

    def __init__(
        self,
        type: RemiEventType,
        position=None,
        instrument=None,
        original_events=[],
    ):
        self.type = type
        self.position = position
        self.instrument = instrument
        self.original_events = original_events

    def __repr__(self):
        return 'Item(type={}, position={}, instrument={})'.format(
            self.type, self.position, self.instrument)


class EfficientRemiConverter(object):

    def __init__(self):
        pass

    def convert_to_remi_items(self, events):
        items = []
        for index, event in enumerate(events):
            if event.name == 'Bar':
                items.append(RemiItem(type=RemiEventType.BAR, original_events=[event]))
            elif event.name == 'Position' and len(events) > index + 2 and events[index + 1].name == 'Tempo Class':
                original_events = [event, events[index + 1], events[index + 2]]
                position = int(event.value.partition("/")[0])
                items.append(RemiItem(type=RemiEventType.TEMPO, position=position, original_events=original_events))
            elif event.name == 'Position' and len(events) > index + 4 and events[index + 1].name == 'Instrument':
                position_event = event
                instrument_event = events[index + 1]
                velocity_event = events[index + 2]
                pitch_event = events[index + 3]
                duration_event = events[index + 4]
                original_events = [position_event, instrument_event, velocity_event, pitch_event, duration_event]
                position = int(position_event.value.partition("/")[0])
                items.append(RemiItem(
                    type=RemiEventType.NOTE,
                    position=position,
                    instrument=instrument_event.value,
                    original_events=original_events)
                )

        items_split_on_bars = []
        current_bar_items = []

        for item in items[1:]:
            if item.type == RemiEventType.BAR:
                items_split_on_bars.append(current_bar_items)
                current_bar_items = []
            else:
                current_bar_items.append(item)

        if len(current_bar_items) > 0:
            items_split_on_bars.append(current_bar_items)

        result = []
        for bar_items in items_split_on_bars:
            result.append(RemiItem(type=RemiEventType.BAR, original_events=[Event(name='Bar', value='None', text='1', time=0)]))
            result.extend(self.sort_bar_items(bar_items))

        return result

    def convert_to_efficient_remi(self, events):
        remi_items = self.convert_to_remi_items(events)
        remi_events = self.convert_back_to_events(remi_items)
        return list(map(lambda x: '{}_{}'.format(x.name, x.value), remi_events))

    def convert_to_normal_remi(self, efficient_remi_words):
        result = []
        last_instrument = "0"
        last_position = "1/16"

        for index, word in enumerate(efficient_remi_words):
            word = efficient_remi_words[index]

            if word.startswith('Instrument'):
                last_instrument = word.split("_")[1]

            if word.startswith('Position'):
                last_position = word.split("_")[1]

            if word.startswith('Note Velocity'):
                if not result[-1].startswith('Instrument'):
                    result.append(f'Instrument_{last_instrument}')
                if not result[-2].startswith('Position'):
                    result.insert(-1, f'Position_{last_position}')

            result.append(word)

        return result

    def sort_bar_items(self, bar_items):
        bar_items.sort(key=lambda x: (x.type, x.instrument, x.position))
        return bar_items

    def convert_back_to_events(self, remi_items):
        events = []
        current_instrument = None
        current_position = None
        for item in remi_items:
            if item.type == RemiEventType.BAR:
                events.extend(item.original_events)
                current_instrument = None
                current_position = None
            elif item.type == RemiEventType.TEMPO:
                events.extend(item.original_events)
            else:
                write_instrument = False
                write_position = False

                if current_instrument is None or current_instrument != item.instrument:
                    current_instrument = item.instrument
                    write_instrument = True
                    write_position = True

                if current_position is None or current_position != item.position:
                    current_position = item.position
                    write_position = True

                if write_position:
                    events.append(item.original_events[0])
                if write_instrument:
                    events.append(item.original_events[1])
                events.extend(item.original_events[2:])

        return events
