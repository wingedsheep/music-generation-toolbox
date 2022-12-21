# Music Generation Toolbox

State-of-the-art algorithms for generation music.

# Usage

See the examples folder for more examples.

### Transformer

A transformer model can be trained on a collection of MIDI files to produce similar sounding music.

```python
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.models.transformer_model import TransformerModel

import os
import glob

# Collect midi paths
midi_path = 'YOUR MIDI PATH'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

# Create datamanager and prepare the data
datamanager = RemiDataManager()
dataset = datamanager.prepare_data(midis)

# Create and train the model
model = TransformerModel(
    dictionary=dataset.dictionary,
    max_sequence_length=512
)

model.train(
    x_train=dataset.data, 
    epochs=50, 
    stop_loss=0.1
)

# Generate music
output = model.generate(1000)

# Restore MIDI file from output and save it
midi = datamanager.to_midi(output)
midi.save("result.midi")
```

Saving and loading a model can be done as follows.
All models have the `save_checkpoint` and `load_checkpoint` functionality.

```python
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.models.transformer_model import TransformerModel

datamanager = RemiDataManager()

model = TransformerModel(
    dictionary=datamanager.dictionary,
    max_sequence_length=512
)

model.save_checkpoint('saved_model')
model2 = TransformerModel.load_checkpoint('saved_model')
```

### Routing Transformer

The Routing Transformer (https://arxiv.org/abs/2003.05997) is an efficient transformer model that can be used to increase the size of the attention window.

```python
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.models.routing_transformer_model import RoutingTransformerModel

import os
import glob

# Collect midi paths
midi_path = 'YOUR MIDI PATH'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

# Create datamanager and prepare the data
datamanager = RemiDataManager()
dataset = datamanager.prepare_data(midis)

# Create and train the model
model = RoutingTransformerModel(
    dictionary=dataset.dictionary,
    max_sequence_length=8192
)

model.train(
    x_train=dataset.data, 
    epochs=50, 
    stop_loss=0.1
)

# Generate music
output = model.generate(1000)

# Restore MIDI file from output and save it
midi = datamanager.to_midi(output)
midi.save("result.midi")
```

### Compound Word Transformer

The compound word transformer (https://arxiv.org/abs/2101.02402) can be used to process midi data more efficiently than REMI. 
Groups of tokens that are always used together are combined into compound words. Every note in REMI consists of: pitch, instrument, duration, velocity and position. 
So these 5 words are combined into a single compound word, that is mapped to a single embedding. This means that effectively every note can now be expressed just with 1 token, instead of 5, giving the advantage of a larger attention window.

Under the hood the CompoundWordTransformerModel uses a standard transformer.

Note that the `CompoundWordTransformer` can only be used with a dataset prepared by the `CompoundWordDataManager`.

```python
from mgt.datamanagers.compound_word_data_manager import CompoundWordDataManager
from mgt.models.compound_word_transformer_model import CompoundWordTransformerModel

import os
import glob

# Collect midi paths
midi_path = 'YOUR MIDI PATH'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

# Create datamanager and prepare the data
datamanager = CompoundWordDataManager()
dataset = datamanager.prepare_data(midis)

# Create and train the model
model = CompoundWordTransformerModel(
    max_sequence_length=512
)

model.train(
    x_train=dataset.data, 
    epochs=50, 
    stop_loss=0.1
)

# Generate music
output = model.generate(1000)

# Restore MIDI file from output and save it
midi = datamanager.to_midi(output)
midi.save("result.midi")
```

### Efficient REMI

Efficient REMI is introduced to fit REMI encoded MIDI in a shorter sequence. Instead of adding an instrument and position token for every note, 
instrument tokens are only added once for every measure and position tokens are only added when the position is different than the previous position. 
For Lakh data this reduces the sequence length for a song by around 25%. For further reduction velocity (volume) information can be removed decreasing the length by more than 40%.

```python
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.datamanagers.remi.efficient_remi_config import EfficientRemiConfig

import os
import glob

# Collect midi paths
midi_path = 'YOUR MIDI PATH'
midi_path = os.path.join(os.path.dirname(__file__), midi_path)
midis = glob.glob(midi_path + '*.mid')

# Create datamanager and prepare the data
datamanager = RemiDataManager(
    efficient_remi_config=EfficientRemiConfig(enabled=True, remove_velocity=True)
)

efficient_remi_data = datamanager.prepare_data(midis)

# Restore MIDI file from output and save it
midi = datamanager.to_midi(efficient_remi_data.data[0])
midi.save("result.midi")
```

# Samples

* Compound word transformer trained on pop909: https://soundcloud.com/user-419192262-663004693/sets/compound-word-transformer-pop909
* Routing transformer trained on pop909: https://soundcloud.com/user-419192262-663004693/sets/routing-transformer-pop909
* Transformer trained on Lakh midi dataset: https://soundcloud.com/user-419192262-663004693/sets/generated-by-music-transformer-from-scratch

# Thanks to 

Great transformers implementations from lucidrains
* https://github.com/lucidrains/reformer-pytorch
* https://github.com/lucidrains/x-transformers
* https://github.com/lucidrains/routing-transformer

Pop music transformer and REMI format
* https://github.com/YatingMusic/remi

Compound word transformer
* https://github.com/YatingMusic/compound-word-transformer

Pop909 dataset
* https://github.com/music-x-lab/POP909-Dataset

Lakh midi dataset
* https://colinraffel.com/projects/lmd/

# Issues

There are still some issues with the reformer model implementation.
It often does not learn how to generate the beginning of the songs well.
This is still a work in progress.
