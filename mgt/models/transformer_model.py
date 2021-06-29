from __future__ import annotations
from datetime import time

import random
import time

import torch
import numpy as np

from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper

from mgt.datamanagers.data_manager import Dictionary


def create_chunks(iterable, chunk_size=1):
    array_length = len(iterable)
    for ndx in range(0, array_length, chunk_size):
        yield iterable[ndx:min(ndx + chunk_size, array_length)]


def create_sequences(training_data, max_sequence_length, padding_character=0):
    sequences = []
    for song in training_data:
        padded_song = list(np.repeat([padding_character], max_sequence_length - 1)) + song
        for i in range(len(padded_song) - max_sequence_length):
            sequence = padded_song[i: i + max_sequence_length]
            sequences.append(sequence)
    return sequences


def pad_training_data(training_data, max_sequence_length, padding_character=0):
    padded_training_data = []
    for song in training_data:
        padded_song = list(np.repeat([padding_character], max_sequence_length)) + song
        padded_training_data.append(padded_song)
    return padded_training_data


def get_batches(padded_training_data, batches_per_epoch, batch_size, max_sequence_length):
    indices = []
    for i in range(batches_per_epoch * batch_size):
        song_index = random.randint(0, len(padded_training_data) - 1)
        starting_index = random.randint(0, len(padded_training_data[song_index]) - max_sequence_length - 1)
        indices.append((song_index, starting_index))

    sequences = []
    for selection in indices:
        sequences.append(padded_training_data[selection[0]][selection[1]: selection[1] + max_sequence_length + 1])

    return list(create_chunks(sequences, chunk_size=batch_size))


class TransformerModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 max_sequence_length=512,
                 learning_rate=2e-4,
                 dropout=0.2,
                 dim=512,
                 depth=12,
                 heads=8
                 ):
        self.dictionary = dictionary
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.model = self.create_model(
            num_tokens=dictionary.size(),
            max_seq_len=max_sequence_length,
            dropout=dropout,
            dim=dim,
            depth=depth,
            heads=heads
        )
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=0.1, batches_per_epoch=100, report_per_x_batches=20):
        self.model.train()
        start_time = time.time()
        padded_training_data = pad_training_data(x_train, self.max_sequence_length)
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")

            batches = get_batches(
                padded_training_data,
                batches_per_epoch=batches_per_epoch,
                batch_size=batch_size,
                max_sequence_length=self.max_sequence_length)

            epoch_losses = []
            batch_losses = []
            nr_of_batches_processed = 0
            for batch in batches:
                torch_batch = torch.tensor(batch).long()

                if torch.cuda.is_available():
                    torch_batch.cuda()

                loss = self.model(torch_batch)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

                nr_of_batches_processed += 1

                loss_item = loss.item()

                batch_losses.append(loss_item)
                epoch_losses.append(loss_item)

                if nr_of_batches_processed % report_per_x_batches == 0:
                    print(f"Processed {nr_of_batches_processed} / {len(batches)} with loss {np.mean(batch_losses)}.")
                    batch_losses = []

            epoch_loss = np.mean(epoch_losses)
            if epoch_loss <= stop_loss:
                print(f"Loss of {epoch_loss} was lower than stop loss of {stop_loss}. Stopping training.")
                return

            running_time = (time.time() - start_time)
            print(f"Loss after epoch {epoch + 1} is {epoch_loss}. Running time: {running_time}")

    def generate(self, output_length=100, temperature=1., filter_treshold=0.9, prompt=None):
        if prompt is None:
            prompt = [0]

        self.model.eval()
        initial = torch.tensor([prompt]).long()  # assume 0 is start token

        if torch.cuda.is_available():
            initial.cuda()

        sample = self.model.generate(initial, output_length, temperature=temperature, filter_thres=filter_treshold)
        return sample.cpu().detach().numpy()[0]

    def create_model(self, num_tokens, max_seq_len, dropout, dim, depth, heads):
        model = AutoregressiveWrapper(TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                attn_dropout=dropout,  # dropout post-attention
                ff_dropout=dropout,  # feedforward dropout
                rotary_pos_emb=True
            )
        ),
            ignore_index=0,
            pad_value=0

        )

        if torch.cuda.is_available():
            model.cuda()

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'dictionary': self.dictionary,
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
    def load_checkpoint(path) -> TransformerModel:
        checkpoint = torch.load(path)
        model = TransformerModel(
            checkpoint['dictionary'],
            checkpoint['max_sequence_length'],
            checkpoint['learning_rate'],
            checkpoint['dropout'],
            checkpoint['dim'],
            checkpoint['depth'],
            checkpoint['heads']
        )

        if torch.cuda.is_available():
            model.cuda()

        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model
