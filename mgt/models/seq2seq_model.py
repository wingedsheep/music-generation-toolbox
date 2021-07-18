from __future__ import annotations
from datetime import time

import random
import time

import torch
import numpy as np

from x_transformers import XTransformer

from mgt.datamanagers.data_manager import Dictionary


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_batch(sources, targets, batch_size):
    indices = random.sample(list(range(len(sources))), batch_size)
    return [sources[i] for i in indices], [targets[i] for i in indices]


class Seq2seqModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 max_input_sequence_length=512,
                 max_output_sequence_length=512,
                 learning_rate=1e-4,
                 dropout=0.0,
                 dim=512,
                 depth=4,
                 heads=3
                 ):
        self.dictionary = dictionary
        self.learning_rate = learning_rate
        self.max_input_sequence_length = max_input_sequence_length
        self.max_output_sequence_length = max_output_sequence_length
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def train(self, sources, targets, epochs, batch_size=4, stop_loss=None, batches_per_epoch=100, report_per_x_batches=20):
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")

            epoch_losses = []
            batch_losses = []
            nr_of_batches_processed = 0
            for _ in range(batches_per_epoch):
                batch_sources, batch_targets = get_batch(
                    sources,
                    targets,
                    batch_size=batch_size)

                batch_sources = torch.tensor(batch_sources).long().to(get_device())
                batch_targets = torch.tensor(batch_targets).long().to(get_device())
                src_mask = torch.ones(batch_size, batch_sources.shape[1]).bool().to(get_device())
                tgt_mask = torch.ones(batch_size, batch_targets.shape[1]).bool().to(get_device())

                loss = self.model(batch_sources, batch_targets, src_mask, tgt_mask)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad()

                nr_of_batches_processed += 1

                loss_item = loss.item()

                batch_losses.append(loss_item)
                epoch_losses.append(loss_item)

                if nr_of_batches_processed % report_per_x_batches == 0:
                    print(f"Processed {nr_of_batches_processed} / {len(batch_sources)} with loss {np.mean(batch_losses)}.")
                    batch_losses = []

            epoch_loss = np.mean(epoch_losses)
            if stop_loss is not None and epoch_loss <= stop_loss:
                print(f"Loss of {epoch_loss} was lower than stop loss of {stop_loss}. Stopping training.")
                return

            running_time = (time.time() - start_time)
            print(f"Loss after epoch {epoch + 1} is {epoch_loss}. Running time: {running_time}")

    def generate(self, sequence_in, sequence_out_start=None, max_output_length=100, eos_character=None):
        if sequence_out_start is None:
            sequence_out_start = [0]

        self.model.eval()
        seq_in = torch.tensor([sequence_in]).long().to(get_device())

        result = []
        if eos_character is not None:
            while len(result) < max_output_length:
                start_sequence = sequence_out_start
                start_sequence.extend(result)
                start_sequence = start_sequence[-self.max_output_sequence_length:]
                start_sequence = torch.tensor([start_sequence]).long().to(get_device())
                seq_in = torch.tensor([sequence_in]).long().to(get_device())
                sample = self.model.generate(seq_in, start_sequence, seq_len=1)
                sample = sample.cpu().detach().numpy()[0]
                if sample[0] == eos_character:
                    return result
                else:
                    result.append(sample[0])
        else:
            seq_out_start = torch.tensor([sequence_out_start]).long().to(get_device())
            sample = self.model.generate(seq_in, seq_out_start, seq_len=max_output_length)
            result = sample.cpu().detach().numpy()[0]

        return result

    def create_model(self):
        return XTransformer(
            dim=self.dim,
            tie_token_embeds=True,
            return_tgt_loss=True,
            enc_num_tokens=self.dictionary.size(),
            enc_depth=self.depth,
            enc_heads=self.heads,
            enc_max_seq_len=self.max_input_sequence_length,
            dec_num_tokens=self.dictionary.size(),
            dec_depth=self.depth,
            dec_heads=self.heads,
            dec_max_seq_len=self.max_output_sequence_length
        ).to(get_device())

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'dictionary': self.dictionary,
            'max_input_sequence_length': self.max_input_sequence_length,
            'max_output_sequence_length': self.max_output_sequence_length,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path) -> Seq2seqModel:
        checkpoint = torch.load(path)
        model = Seq2seqModel(
            dictionary=checkpoint['dictionary'],
            max_input_sequence_length=checkpoint['max_input_sequence_length'],
            max_output_sequence_length=checkpoint['max_output_sequence_length'],
            learning_rate=checkpoint['learning_rate'],
            dropout=checkpoint['dropout'],
            dim=checkpoint['dim'],
            depth=checkpoint['depth'],
            heads=checkpoint['heads']
        )

        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return model
