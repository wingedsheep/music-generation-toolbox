from mgt.datamanagers.time_shift_data_manager import TimeShiftDataManager
from mgt.models.routing_transformer_model import RoutingTransformerModel

import os
import glob


def run():
    """
    Example showing how to train a new model and generate new music with it.
    """

    midi_path = '../data/pop/'
    midi_path = os.path.join(os.path.dirname(__file__), midi_path)
    midis = glob.glob(midi_path + '*.mid')

    time_shift_data_manager = TimeShiftDataManager()
    dataset = time_shift_data_manager.prepare_data(midis)

    model = RoutingTransformerModel(dataset.dictionary)

    print("Created model. Starting training for 50 epochs.")
    model.train(x_train=dataset.data, epochs=50, stop_loss=0.1)

    # Generate music
    print("Generating music.")
    output = model.generate(1000)

    # Restore events from input data
    midi = time_shift_data_manager.to_midi(output)
    midi.save("result.midi")


run()
