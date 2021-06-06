import string

from encoders.event_extractor import Event


class WordsConverter(object):

    @staticmethod
    def events_to_words(events: [Event]):
        words = []
        current_program = None
        current_velocity = None
        for event in events:
            if event.event_type == "start-track":
                words.append("start-track")
            elif event.event_type == "end-track":
                words.append("end-track")
            elif event.event_type == "time-shift":
                duration = event.data['duration']
                while duration > 128:
                    words.append(f"time-shift_{min(128, duration)}")
                    duration -= 128
                words.append(f"time-shift_{duration}")
            elif event.event_type == "note":
                if event.data["program"] != current_program:
                    words.append(f"program_{event.data['program']}")
                    current_program = event.data['program']
                if event.data["velocity"] != current_velocity:
                    # Divide by four because we put the velocity into 32 bins
                    words.append(f"velocity_{int(event.data['velocity'] / 4)}")
                    current_velocity = event.data['velocity']
                words.append(f"note_{event.data['pitch']}")
                duration = event.data['duration'] if event.data['duration'] > 0 else 1
                while duration > 128:
                    words.append(f"duration_{min(128, duration)}")
                    duration -= 128
                words.append(f"duration_{duration}")

        return words

    @staticmethod
    def words_to_events(words: [string]):
        events = []
        current_time = 0
        current_instrument = 0
        current_velocity = 12
        for index in range(len(words)):
            word = words[index]
            if "time-shift" in word:
                duration = int(word.split("_")[1])
                current_time += duration
            elif "program" in word:
                program = int(word.split("_")[1])
                current_instrument = program
            elif "velocity" in word:
                velocity = int(word.split("_")[1])
                current_velocity = velocity
            elif "note" in word:
                pitch = int(word.split("_")[1])
                start = current_time
                program = current_instrument
                velocity = current_velocity
                duration = 0
                while "duration" in words[index + 1]:
                    index += 1
                    word = words[index]
                    extracted_duration = int(word.split("_")[1])
                    duration += extracted_duration
                events.append(Event(
                    event_type="note",
                    start=start,
                    data={
                        "program": program,
                        "velocity": velocity * 4,  # Multiply velocity by 4, because we used bins of 4
                        "duration": duration,
                        "pitch": pitch
                    }
                ))
        return events
