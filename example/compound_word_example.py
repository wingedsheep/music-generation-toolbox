import glob
import os

from mgt.datamanagers.compound_word_data_manager import CompoundWordDataManager
from mgt.models.compound_word_transformer_model import CompoundWordTransformerModel

data_manager = CompoundWordDataManager()
midi_path = '../data/pop/'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

x_train = data_manager.prepare_data(midis)
print(x_train.data)

model = CompoundWordTransformerModel(
    num_tokens=[
        4,    # Type
        17,   # Bar / Beat
        192,  # Tempo
        129,  # Instrument
        128,  # Pitch
        64,   # Duration
        32    # Velocity
    ],
    max_sequence_length=2
)
model.train(x_train.data, epochs=1)
