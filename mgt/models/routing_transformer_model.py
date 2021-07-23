from __future__ import annotations
from datetime import time

import random
import time

import torch
import numpy as np

from mgt.datamanagers.data_manager import Dictionary
from routing_transformer import RoutingTransformerLM, AutoregressiveWrapper


def pad(array, max_sequence_length, padding_character=0):
    return list(np.repeat(padding_character, max_sequence_length)) + array


def get_batch(training_data, batch_size, max_sequence_length):
    indices = []
    for i in range(batch_size):
        song_index = random.randint(0, len(training_data) - 1)
        starting_index = random.randint(0, len(training_data[song_index]) - 1)
        indices.append((song_index, starting_index))

    sequences = []
    for selection in indices:
        padded_song = pad(training_data[selection[0]], max_sequence_length)
        sequences.append(padded_song[selection[1]: selection[1] + max_sequence_length + 1])

    return sequences


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RoutingTransformerModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 max_sequence_length=4096,
                 learning_rate=1e-4,
                 dropout=0.1,
                 dim=512,
                 depth=6,
                 heads=4,
                 window_size=256
                 ):
        self.dictionary = dictionary
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.window_size = window_size
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=None, batches_per_epoch=100, report_per_x_batches=20):
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")

            epoch_losses = []
            batch_losses = []
            nr_of_batches_processed = 0
            for _ in range(batches_per_epoch):
                batch = get_batch(
                    x_train,
                    batch_size=batch_size,
                    max_sequence_length=self.max_sequence_length)

                torch_batch = torch.tensor(batch).long().to(get_device())

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
                    print(f"Processed {nr_of_batches_processed} / {batches_per_epoch} with loss {np.mean(batch_losses)}.")
                    batch_losses = []

            epoch_loss = np.mean(epoch_losses)
            if stop_loss is not None and epoch_loss <= stop_loss:
                print(f"Loss of {epoch_loss} was lower than stop loss of {stop_loss}. Stopping training.")
                return

            running_time = (time.time() - start_time)
            print(f"Loss after epoch {epoch + 1} is {epoch_loss}. Running time: {running_time}")

    def generate(self, output_length=100, temperature=1., filter_treshold=0.9, prompt=None):
        print(f"Generating a new song with {output_length} characters.")
        if prompt is None:
            prompt = [0]

        self.model.eval()
        initial = torch.tensor([prompt]).long().to(get_device())  # assume 0 is start token

        sample = self.model.generate(initial, output_length, temperature=temperature, filter_thres=filter_treshold)
        return sample.cpu().detach().numpy()[0]

    def create_model(self):
        model = RoutingTransformerLM(
            num_tokens=self.dictionary.size(),
            dim=self.dim,
            heads=self.heads,
            depth=self.depth,
            window_size=self.window_size,
            max_seq_len=self.max_sequence_length,
            attn_dropout=self.dropout,
            ff_dropout=self.dropout,
            causal=True
        )

        model = AutoregressiveWrapper(model,
                                      ignore_index=0,
                                      pad_value=0
                                      ).to(get_device())

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
            'window_size': self.window_size,
            'heads': self.heads,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path) -> RoutingTransformerModel:
        checkpoint = torch.load(path)
        model = RoutingTransformerModel(
            dictionary=checkpoint['dictionary'],
            max_sequence_length=checkpoint['max_sequence_length'],
            learning_rate=checkpoint['learning_rate'],
            dropout=checkpoint['dropout'],
            dim=checkpoint['dim'],
            depth=checkpoint['depth'],
            window_size=checkpoint['window_size'],
            heads=checkpoint['heads']
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model
