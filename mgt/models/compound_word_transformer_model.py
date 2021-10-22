from __future__ import annotations

import random
import time

import numpy as np
import torch
from x_transformers import Decoder

from mgt.models.compound_word_transformer.compound_word_autoregressive_wrapper import CompoundWordAutoregressiveWrapper
from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_BAR, pad, \
    COMPOUND_WORD_PADDING
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device


def get_batch(training_data, batch_size, max_sequence_length, randomly_truncate=False):
    indices = []
    for i in range(batch_size):
        song_index = random.randint(0, len(training_data) - 1)
        starting_index = random.randint(0, len(training_data[song_index]) - 1)
        indices.append((song_index, starting_index))

    sequences = []
    for selection in indices:
        padded_song = pad(training_data[selection[0]], max_sequence_length + len(training_data[selection[0]]))
        if randomly_truncate:
            rand_length = random.randint(0, max_sequence_length - 2)
            for i in range(rand_length):
                padded_song[i] = COMPOUND_WORD_PADDING
        sequences.append(padded_song[selection[1]: selection[1] + max_sequence_length + 1])

    return sequences


class CompoundWordTransformerModel(object):

    def __init__(self,
                 num_tokens=None,
                 emb_sizes=None,
                 max_sequence_length=512,
                 learning_rate=1e-4,
                 dropout=0.1,
                 dim=512,
                 depth=12,
                 heads=8
                 ):
        if num_tokens is None:
            num_tokens = [
                4,  # Type
                17,  # Bar / Beat
                192,  # Tempo
                129,  # Instrument
                128,  # Pitch
                64,  # Duration
                32  # Velocity
            ]
        self.num_tokens = num_tokens
        self.emb_sizes = emb_sizes
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = self.create_optimizer()

    def train(self,
              x_train,
              epochs,
              batch_size=4,
              stop_loss=None,
              batches_per_epoch=100,
              report_per_x_batches=20,
              randomly_truncate=True,
              gradient_accumulation_steps=1):
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")

            epoch_losses = []
            batch_losses = []
            nr_of_batches_processed = 0
            for _ in range(batches_per_epoch):
                for _ in range(gradient_accumulation_steps):
                    batch = get_batch(
                        x_train,
                        batch_size=batch_size,
                        max_sequence_length=self.max_sequence_length,
                        randomly_truncate=randomly_truncate)

                    torch_batch = torch.tensor(batch).long().to(get_device())

                    losses = self.model.train_step(torch_batch)
                    loss = (losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5] + losses[6]) / 7
                    loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

                nr_of_batches_processed += 1

                loss_item = loss.item()

                batch_losses.append(loss_item)
                epoch_losses.append(loss_item)

                if nr_of_batches_processed % report_per_x_batches == 0:
                    print(
                        f"Processed {nr_of_batches_processed} / {batches_per_epoch} with loss {np.mean(batch_losses)}.")
                    batch_losses = []

            epoch_loss = np.mean(epoch_losses)
            if stop_loss is not None and epoch_loss <= stop_loss:
                print(f"Loss of {epoch_loss} was lower than stop loss of {stop_loss}. Stopping training.")
                return

            running_time = (time.time() - start_time)
            print(f"Loss after epoch {epoch + 1} is {epoch_loss}. Running time: {running_time}")

    def generate(self, output_length=100, prompt=None):
        print(f"Generating a new song with {output_length} characters.")

        if prompt is None:
            prompt = [COMPOUND_WORD_BAR]  # Bar

        self.model.eval()
        sample = self.model.generate(output_length=output_length, prompt=prompt)
        return sample

    def create_model(self):
        model = CompoundWordAutoregressiveWrapper(CompoundWordTransformerWrapper(
            num_tokens=self.num_tokens,
            emb_sizes=self.emb_sizes,
            max_seq_len=self.max_sequence_length,
            attn_layers=Decoder(
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.dropout,  # dropout post-attention
                ff_dropout=self.dropout,  # feedforward dropout
                rotary_pos_emb=True
            )
        )).to(get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'num_tokens': self.num_tokens,
            'emb_sizes': self.emb_sizes,
            'max_sequence_length': self.max_sequence_length,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path) -> CompoundWordTransformerModel:
        checkpoint = torch.load(path)
        model = CompoundWordTransformerModel(
            num_tokens=checkpoint['num_tokens'],
            emb_sizes=checkpoint['emb_sizes'],
            max_sequence_length=checkpoint['max_sequence_length'],
            learning_rate=checkpoint['learning_rate'],
            dropout=checkpoint['dropout'],
            dim=checkpoint['dim'],
            depth=checkpoint['depth'],
            heads=checkpoint['heads']
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model
