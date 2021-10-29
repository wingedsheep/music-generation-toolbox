# music-generation-tools
Toolbox for generating music

# usage

See the examples folder for more examples.

Training your model and generating music:

```
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
model = TransformerModel(dataset.dictionary)
model.train(x_train=dataset.data, epochs=50, stop_loss=0.1)

# Generate music
output = model.generate(1000)

# Restore MIDI file from output and save it
midi = datamanager.to_midi(output)
midi.save("result.midi")
```

# samples

* Compound word transformer trained on pop909: https://soundcloud.com/user-419192262-663004693/sets/compound-word-transformer-pop909
* Routing transformer trained on pop909: https://soundcloud.com/user-419192262-663004693/sets/routing-transformer-pop909
* Transformer trained on Lakh midi dataset: https://soundcloud.com/user-419192262-663004693/sets/generated-by-music-transformer-from-scratch

# thanks to 

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
