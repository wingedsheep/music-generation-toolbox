from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.models.transformer_model import TransformerModel

import os
import glob

# Collect the midi paths
midi_path = 'YOUR MIDI PATH'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

# Create the datamanager and prepare the data
datamanager = RemiDataManager()
dataset = datamanager.prepare_data(midis)

# Create and train the model
model = TransformerModel(dataset.dictionary)
model.train(x_train=dataset.data, epochs=50, stop_loss=0.1)

# Generate music
output = model.generate(1000)

# Restore events from input data
midi = datamanager.to_midi(output)
midi.save("result.midi")