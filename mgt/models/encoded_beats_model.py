import random
import time

import torch
from x_transformers import Decoder, ContinuousTransformerWrapper, ContinuousAutoregressiveWrapper
import numpy as np

from mgt.datamanagers.encoded_beats.beat_data_auto_encoder import BeatDataAutoEncoder
from mgt.models import utils

defaults = {
    'auto_encoder': BeatDataAutoEncoder(),
    'max_sequence_length': 512,
    'vector_dimension': 512,
    'embedding_dimension': 512,
    'dropout': 0.1,
    'depth': 12,
    'heads': 6,
    'learning_rate': 1e-4
}


def pad_right(song_beat_vectors, max_sequence_length, padding_vector):
    padded_song_beat_vectors = song_beat_vectors.copy()
    while len(padded_song_beat_vectors) < max_sequence_length:
        padded_song_beat_vectors = np.append(padded_song_beat_vectors, [padding_vector], 0)
    return padded_song_beat_vectors


def get_batch(x_train, batch_size, max_sequence_length, padding_vector):
    indices = []
    for i in range(batch_size):
        song_index = random.randint(0, len(x_train) - 1)
        starting_index = random.randint(0, max(0, len(x_train[song_index]) - 1))
        indices.append((song_index, starting_index))

    return np.array(list(map(
        lambda index: pad_right(x_train[index[0]][index[1]:], max_sequence_length, padding_vector),
        indices
    )))


class EncodedBeatsModel(object):

    def __init__(
            self,
            auto_encoder: BeatDataAutoEncoder = defaults['auto_encoder'],
            max_sequence_length: int = defaults['max_sequence_length'],
            vector_dimension: int = defaults['vector_dimension'],
            embedding_dimension: int = defaults['embedding_dimension'],
            dropout: float = defaults['dropout'],
            depth: int = defaults['depth'],
            heads: int = defaults['heads'],
            learning_rate: float = defaults['learning_rate']
    ):
        self.auto_encoder = auto_encoder
        self.max_sequence_length = max_sequence_length
        self.vector_dimension = vector_dimension
        self.embedding_dimension = embedding_dimension
        self.dropout = dropout
        self.depth = depth
        self.heads = heads
        self.learning_rate = learning_rate
        self.padding_vector = np.repeat(-1, vector_dimension)
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=None, batches_per_epoch=1, report_per_x_batches=1,
              gradient_accumulation_steps=1):

        train_vectors = self.auto_encoder.encode(x_train)

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
                        train_vectors,
                        batch_size=batch_size,
                        max_sequence_length=self.max_sequence_length,
                        padding_vector=self.padding_vector
                    )

                    torch_batch = torch.tensor(batch).float().to(utils.get_device())

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

    def generate(self, output_length=20, prompt=None):
        print(f"Generating a new song with {output_length} beats.")
        if prompt is None:
            prompt = [self.padding_vector]

        self.model.eval()
        initial = torch.tensor(prompt).float().to(utils.get_device())

        encoded_sample = self.model.generate(initial, output_length)
        encoded_sample = encoded_sample.cpu().detach().numpy()

        decoded_sample = [self.auto_encoder.decode(x) for x in encoded_sample]

        return decoded_sample

    def create_model(self):
        model = ContinuousAutoregressiveWrapper(ContinuousTransformerWrapper(
            max_seq_len=self.max_sequence_length,
            dim_in=self.vector_dimension,
            dim_out=self.vector_dimension,
            emb_dim=self.embedding_dimension,
            use_pos_emb=True,
            attn_layers=Decoder(
                dim=self.embedding_dimension,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.dropout,
                ff_dropout=self.dropout,
                rotary_pos_emb=True
            )
        ),
            ignore_index=0,
            pad_value=self.padding_vector
        ).to(utils.get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
