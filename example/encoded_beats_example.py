import glob
import os

from mgt.datamanagers.encoded_beats.beat_data_auto_encoder import BeatDataAutoEncoder
from mgt.datamanagers.encoded_beats.encoded_beats_dataset import EncodedBeatsDataSet
from mgt.datamanagers.encoded_beats_data_manager import EncodedBeatsDataManager
from mgt.models.encoded_beats_model import EncodedBeatsModel

"""
Example showing how to train an encoded beats model and generate music with it
"""

midi_path = '../data/pop/'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

# Create a data manager for tracks with three piano (midi program 0) instruments
data_manager = EncodedBeatsDataManager(
    instruments=[0, 0, 0]
)

# Prepare the dataset
dataset: EncodedBeatsDataSet = data_manager.prepare_data(midis)

# Create the auto encoder
auto_encoder = BeatDataAutoEncoder(
    input_dim=dataset.get_encoded_beat_size()
)

# Create the transformer model
model = EncodedBeatsModel(
    auto_encoder=auto_encoder
)

# Train the auto encoder first and then the transformer model
model.auto_encoder.train(dataset.data, epochs=1024)
model.train(dataset.data, epochs=512)

# Generate a song
output = model.generate(200)

# Save the resulting midi
midi = data_manager.to_midi(output)
midi.save("result.midi")
