from mgt.datamanagers.time_shift_data_manager import TimeShiftDataManager
import os
import glob

# Load MIDI files
from mgt.models.reformer_model import ReformerModel


def run():
    """
    Example showing how we can train a new model and generate new music with it.
    """

    midi_path = '../data/pop/'
    midi_path = os.path.join(os.path.dirname(__file__), midi_path)
    midis = glob.glob(midi_path + '*.mid')

    time_shift_data_manager = TimeShiftDataManager()
    dataset = time_shift_data_manager.prepare_data(midis)

    model = ReformerModel(dataset.dictionary)

    print("Created model. Starting training for 4 epochs.")
    model.train(x_train=dataset.data, epochs=4, stop_loss=0.1)

    # Generate music
    print("Generating music.")
    output = model.generate(100)

    # Restore events from input data
    midi = time_shift_data_manager.to_midi(output)
    midi.save("result.midi")


run()
