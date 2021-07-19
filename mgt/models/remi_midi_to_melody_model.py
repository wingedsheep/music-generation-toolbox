from __future__ import annotations

import numpy as np

from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator
from mgt.models.seq2seq_model import Seq2seqModel
from tqdm import tqdm


class RemiMidiToMelodyModel(object):

    def __init__(self,
                 max_sequence_length=512,
                 learning_rate=1e-4,
                 dropout=0.0,
                 dim=512,
                 depth=4,
                 heads=3,
                 max_measures_per_step=2
                 ):
        self.dictionary = DictionaryGenerator.create_dictionary()
        self.dictionary.append("seq_start")
        self.dictionary.append("seq_end")
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.max_measures_per_step = max_measures_per_step
        self.model = Seq2seqModel(
            dictionary=self.dictionary,
            learning_rate=self.learning_rate,
            max_input_sequence_length=self.max_sequence_length,
            max_output_sequence_length=self.max_sequence_length,
            dropout=self.dropout,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads
        )

    def train(self, sources, targets, epochs, batch_size=4, stop_loss=None, batches_per_epoch=100,
              report_per_x_batches=20):

        if len(sources) != len(targets):
            print(f"Number of sources should match number targets. Sources = {len(sources)}, targets = {len(targets)}")
            return

        inputs = []
        outputs = []
        for i in range(len(sources)):
            prepared_input = self.prepare_data(sources[i])
            inputs.extend(prepared_input)
            outputs.extend(self.prepare_data(targets[i], min_measures=len(prepared_input)))

        max_input_length = max([len(x) for x in inputs])
        max_output_length = max([len(x) for x in outputs])
        if max_input_length > self.max_sequence_length or max_output_length > self.max_sequence_length:
            print(f"Max length of input or output should not exceed max sequence length. Max input length = {max_input_length}, max output length = {max_output_length}")
            return

        self.model.train(inputs, outputs, epochs,
                         batch_size=batch_size,
                         stop_loss=stop_loss,
                         batches_per_epoch=batches_per_epoch,
                         report_per_x_batches=report_per_x_batches,
                         mask_characters=[self.dictionary.word_to_data("pad")])

    def prepare_data(self, remi_data, min_measures=0):
        parts = []
        bars = -1
        current_part = []
        for word in remi_data:
            current_part.append(word)
            if self.dictionary.data_to_word(word) == 'Bar_None':
                bars += 1
                if bars == self.max_measures_per_step:
                    current_part.insert(0, self.dictionary.word_to_data("seq_start"))
                    current_part.append(self.dictionary.word_to_data("seq_end"))
                    parts.append(current_part)
                    bars = 0
                    current_part = [self.dictionary.word_to_data("Bar_None")]

        while len(parts) < min_measures:
            parts.append([
                self.dictionary.word_to_data("seq_start"),
                self.dictionary.word_to_data("Bar_None"),
                self.dictionary.word_to_data("Bar_None"),
                self.dictionary.word_to_data("seq_end")
            ])

        return list(map(lambda x: self.pad(x, self.max_sequence_length), parts))

    def pad(self, sequence, max_sequence_length):
        return sequence + list(np.repeat(0, max(0, max_sequence_length - len(sequence))))

    def convert(self, remi_midi):
        inputs = self.prepare_data(remi_midi)
        converted_song = []

        print("Extracting melody.")

        pbar = tqdm(total=len(inputs))

        progress = 0
        for x in inputs:
            generated = self.model.generate(x, max_output_length=self.max_sequence_length,
                                            eos_character=self.dictionary.word_to_data("seq_end"))

            progress += 1
            pbar.update(progress)

            # Remove unused characters in REMI
            result = list(filter(lambda word:
                                 word != self.dictionary.word_to_data("seq_start") and
                                 word != self.dictionary.word_to_data("seq_end") and
                                 word != self.dictionary.word_to_data("pad")
                                 , generated))

            print([self.dictionary.data_to_word(x) for x in generated])

            # Remove last Bar_None char
            converted_song.extend(result[:-1])

        pbar.close()
        return converted_song
