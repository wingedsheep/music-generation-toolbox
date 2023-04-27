from __future__ import annotations
from datetime import time

import time

import torch
import numpy as np

from perceiver_ar_pytorch import PerceiverAR
from perceiver_ar_pytorch.autoregressive_wrapper import AutoregressiveWrapper

from mgt.datamanagers.data_manager import Dictionary
from mgt.models import utils


defaults = {
    'max_sequence_length': 4096,    # total max sequence length
    'cross_attn_seq_len': 3072,     # the sequence length in which to attend to, but does not undergo self attention (must be less than max_seq_len)
    'cross_attn_dropout': 0.5,      # what percentage of the prefix to dropout during training, in paper they had extensive experimentation to show up to 50% dropout helped prevent overfitting
    'learning_rate': 1e-4,
    'dropout': 0.1,
    'dim': 512,                     # model dimensions
    'depth': 8,                     # model depth
    'heads': 8,                     # attention heads
    'dim_head': 64                  # attention head dimension
}


class PerceiverArModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 max_sequence_length=defaults['max_sequence_length'],
                 cross_attn_seq_len=defaults['cross_attn_seq_len'],
                 cross_attn_dropout=defaults['cross_attn_dropout'],
                 learning_rate=defaults['learning_rate'],
                 dropout=defaults['dropout'],
                 dim=defaults['dim'],
                 depth=defaults['depth'],
                 heads=defaults['heads'],
                 dim_head=defaults['dim_head']
                 ):
        self.dictionary = dictionary
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.cross_attn_seq_len = cross_attn_seq_len
        self.cross_attn_dropout = cross_attn_dropout
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=None, batches_per_epoch=100, report_per_x_batches=20,
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
                    batch = utils.get_batch(
                        x_train,
                        batch_size=batch_size,
                        max_sequence_length=self.max_sequence_length)

                    torch_batch = torch.tensor(np.array(batch)).long().to(utils.get_device())

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
                    print(
                        f"Processed {nr_of_batches_processed} / {batches_per_epoch} with loss {np.mean(batch_losses)}.")
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
            prompt = np.zeros(self.cross_attn_seq_len + 1)
        elif len(prompt) < self.cross_attn_seq_len:
            prompt = utils.pad(prompt, self.cross_attn_seq_len + 1, 0)

        self.model.eval()
        initial = torch.tensor(np.array([prompt])).long().to(utils.get_device())

        sample = self.model.generate(
            start_tokens=initial,
            seq_len=output_length,
            temperature=temperature,
            filter_thres=filter_treshold
        )
        return sample.cpu().detach().numpy()[0]

    def create_model(self):
        model = AutoregressiveWrapper(PerceiverAR(
            num_tokens=self.dictionary.size(),
            dim=self.dim,
            depth=self.depth,
            dim_head=self.dim_head,
            heads=self.heads,
            max_seq_len=self.max_sequence_length,
            cross_attn_seq_len=self.cross_attn_seq_len,
            cross_attn_dropout=self.cross_attn_dropout,
        ),
            pad_value=0
        ).to(utils.get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'dictionary': self.dictionary,
            'max_sequence_length': self.max_sequence_length,
            'cross_attn_seq_len': self.cross_attn_seq_len,
            'cross_attn_dropout': self.cross_attn_dropout,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path) -> PerceiverArModel:
        checkpoint = torch.load(path)

        model = PerceiverArModel(
            dictionary=checkpoint['dictionary'],
            max_sequence_length=utils.get_or_default(checkpoint, 'max_sequence_length', defaults),
            cross_attn_seq_len=utils.get_or_default(checkpoint, 'cross_attn_seq_len', defaults),
            cross_attn_dropout=utils.get_or_default(checkpoint, 'cross_attn_dropout', defaults),
            learning_rate=utils.get_or_default(checkpoint, 'learning_rate', defaults),
            dropout=utils.get_or_default(checkpoint, 'dropout', defaults),
            dim=utils.get_or_default(checkpoint, 'dim', defaults),
            depth=utils.get_or_default(checkpoint, 'depth', defaults),
            heads=utils.get_or_default(checkpoint, 'heads', defaults),
            dim_head=utils.get_or_default(checkpoint, 'dim_head', defaults),
        )

        model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return model
