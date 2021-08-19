import glob
import os

from mgt.datamanagers.compound_word_data_manager import CompoundWordDataManager
from mgt.models.compound_word_transformer_model import CompoundWordTransformerModel


def run():
    """
    Example showing how to train a compound word model and generate new music with it.
    """

    midi_path = '../data/pop/'
    midi_path = os.path.join(os.path.dirname(__file__), midi_path)
    midis = glob.glob(midi_path + '*.mid')

    data_manager = CompoundWordDataManager()
    dataset = data_manager.prepare_data(midis)

    model = CompoundWordTransformerModel()

    print("Created model. Starting training for 50 epochs.")
    model.train(x_train=dataset.data, epochs=50, stop_loss=0.1)

    # Generate music
    print("Generating music.")
    output = model.generate(1000)

    # Restore events from input data
    midi = data_manager.to_midi(output)
    midi.save("result.midi")


run()
