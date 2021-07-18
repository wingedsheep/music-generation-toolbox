from mgt.datamanagers.remi_data_manager import RemiDataManager
import os

from mgt.datamanagers.time_shift_data_manager import TimeShiftDataManager
from mgt.models.remi_midi_to_melody_model import RemiMidiToMelodyModel


def run():
    """
    Example showing how to parse midi files with different encodings.
    Also outputs midi files showing how these encodings sound when converted back.
    """
    input_path = '../data/seq2seq/Miley Cyrus - Wrecking ball.mid'
    input_path = os.path.join(os.path.dirname(__file__), input_path)

    output_path = '../data/seq2seq/Miley Cyrus - Wrecking ball melody.mid'
    output_path = os.path.join(os.path.dirname(__file__), output_path)

    datamanager = RemiDataManager()

    sources = datamanager.prepare_data([input_path])
    targets = datamanager.prepare_data([output_path])

    model = RemiMidiToMelodyModel()
    model.train(
        sources=sources.data,
        targets=targets.data,
        epochs=25,
        batch_size=1)

    result = model.convert(sources.data[0])
    midi = datamanager.to_midi(result)
    midi.save("converted.midi")


run()
