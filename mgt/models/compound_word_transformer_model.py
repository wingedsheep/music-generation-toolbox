from __future__ import annotations

import math
from datetime import time

import random
import time

import numpy as np
from torch.autograd.grad_mode import F

from x_transformers import Decoder
from x_transformers.x_transformers import AttentionLayers, default, AbsolutePositionalEmbedding, always

import torch
from torch import nn
import torch.nn.functional as F
from entmax import entmax_bisect


def pad(array: list, max_sequence_length, padding_character=None):
    if padding_character is None:
        padding_character = [0, 0, 0, 0, 0, 0, 0, 0]
    padded_array = array.copy()
    for _ in range(max_sequence_length):
        padded_array.insert(0, padding_character)
    return padded_array


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


class CompoundTransformerEmbeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(CompoundTransformerEmbeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    def weight(self):
        return self.lut.weight


class CompoundTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True,
            emb_sizes=None
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        if emb_sizes is None:
            emb_sizes = [
                32,  # Type
                64,  # Bar / Beat
                128,  # Tempo
                128,  # Instrument
                384,  # Pitch
                128,  # Duration
                128,  # Velocity

            ]

        self.emb_sizes = emb_sizes

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.num_tokens = num_tokens

        self.max_seq_len = max_seq_len

        self.word_emb_type = CompoundTransformerEmbeddings(self.num_tokens[0], self.emb_sizes[0])
        self.word_emb_barbeat = CompoundTransformerEmbeddings(self.num_tokens[1], self.emb_sizes[1])
        self.word_emb_tempo = CompoundTransformerEmbeddings(self.num_tokens[2], self.emb_sizes[2])
        self.word_emb_instrument = CompoundTransformerEmbeddings(self.num_tokens[3], self.emb_sizes[3])
        self.word_emb_pitch = CompoundTransformerEmbeddings(self.num_tokens[4], self.emb_sizes[4])
        self.word_emb_duration = CompoundTransformerEmbeddings(self.num_tokens[5], self.emb_sizes[5])
        self.word_emb_velocity = CompoundTransformerEmbeddings(self.num_tokens[6], self.emb_sizes[6])

        # individual output
        self.proj_type = nn.Linear(dim, self.num_tokens[0])
        self.proj_barbeat = nn.Linear(dim, self.num_tokens[1])
        self.proj_tempo = nn.Linear(dim, self.num_tokens[2])
        self.proj_instrument = nn.Linear(dim, self.num_tokens[3])
        self.proj_pitch = nn.Linear(dim, self.num_tokens[4])
        self.proj_duration = nn.Linear(dim, self.num_tokens[5])
        self.proj_velocity = nn.Linear(dim, self.num_tokens[6])

        # in_features is equal to dimension plus dimensions of the type embedding
        self.project_concat_type = nn.Linear(dim + self.emb_sizes[0], dim)

        self.compound_word_embedding_size = np.sum(emb_sizes)

        self.pos_emb = AbsolutePositionalEmbedding(self.compound_word_embedding_size, max_seq_len) if (
                use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.in_linear = nn.Linear(np.sum(self.emb_sizes), emb_dim)

        self.init_()

    def init_(self):
        nn.init.normal_(self.word_emb_type.weight(), std=0.02)
        nn.init.normal_(self.word_emb_barbeat.weight(), std=0.02)
        nn.init.normal_(self.word_emb_tempo.weight(), std=0.02)
        nn.init.normal_(self.word_emb_instrument.weight(), std=0.02)
        nn.init.normal_(self.word_emb_pitch.weight(), std=0.02)
        nn.init.normal_(self.word_emb_duration.weight(), std=0.02)
        nn.init.normal_(self.word_emb_velocity.weight(), std=0.02)

    def forward_output(self,
                       h,
                       target
                       ):
        tf_skip_type = self.word_emb_type(target[..., 0])

        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        emb_barbeat = self.proj_barbeat(y_)
        emb_tempo = self.proj_tempo(y_)
        emb_instrument = self.proj_instrument(y_)
        emb_pitch = self.proj_pitch(y_)
        emb_duration = self.proj_duration(y_)
        emb_velocity = self.proj_velocity(y_)

        return emb_barbeat, emb_tempo, emb_instrument, emb_pitch, emb_duration, emb_velocity

    def forward_hidden(
            self,
            x,
            mask=None,
            **kwargs
    ):
        # embeddings
        emb_type = self.word_emb_type(x[..., 0])
        emb_barbeat = self.word_emb_barbeat(x[..., 1])
        emb_tempo = self.word_emb_tempo(x[..., 2])
        emb_instrument = self.word_emb_instrument(x[..., 3])
        emb_pitch = self.word_emb_pitch(x[..., 4])
        emb_duration = self.word_emb_duration(x[..., 5])
        emb_velocity = self.word_emb_velocity(x[..., 6])

        embs = torch.cat(
            [
                emb_type,
                emb_barbeat,
                emb_tempo,
                emb_instrument,
                emb_pitch,
                emb_duration,
                emb_velocity
            ], dim=-1)

        emb_linear = self.in_linear(embs)

        x = emb_linear + self.pos_emb(emb_linear)
        x = self.emb_dropout(x)
        x = self.project_emb(x)

        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
        x = self.norm(x)

        return x, self.proj_type(x)


# nucleus

def top_p(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk

def top_k(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


# entmax

ENTMAX_ALPHA = 1.3
entmax = entmax_bisect


class CompoundWordAutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index=-100, pad_value=None):
        super().__init__()
        if pad_value is None:
            pad_value = [0, 0, 0, 0, 0, 0, 0, 0]
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9,
                 **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres=filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def calculate_loss(self, predicted, target, loss_mask):
        if loss_mask is None:
            loss_mask = torch.ones_like(predicted).bool().to(get_device())
        loss = F.cross_entropy(predicted[:, ...].permute(0, 2, 1), target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, **kwargs):
        xi = x[:, :-1]
        xo = x[:, 1:]

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        h, proj_type = self.net.forward_hidden(xi, **kwargs)
        proj_barbeat, proj_tempo, proj_instrument, proj_pitch, proj_duration, proj_velocity = self.net.forward_output(h, xo)

        type_loss = self.calculate_loss(proj_type, xo[..., 0], mask)
        barbeat_loss = self.calculate_loss(proj_barbeat, xo[..., 1], mask)
        tempo_loss = self.calculate_loss(proj_tempo, xo[..., 2], mask)
        instrument_loss = self.calculate_loss(proj_instrument, xo[..., 3], mask)
        pitch_loss = self.calculate_loss(proj_pitch, xo[..., 4], mask)
        duration_loss = self.calculate_loss(proj_duration, xo[..., 5], mask)
        velocity_loss = self.calculate_loss(proj_velocity, xo[..., 6], mask)

        return type_loss, barbeat_loss, tempo_loss, instrument_loss, pitch_loss, duration_loss, velocity_loss


class CompoundWordTransformerModel(object):

    def __init__(self,
                 num_tokens,  # Number of tokens per category []
                 max_sequence_length=512,
                 learning_rate=1e-4,
                 dropout=0.1,
                 dim=512,
                 depth=12,
                 heads=8
                 ):
        self.num_tokens = num_tokens
        self.learning_rate = learning_rate
        self.max_sequence_length = max_sequence_length
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
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

    def generate(self, output_length=100, temperature=1., filter_treshold=0.9, prompt=None):
        print(f"Generating a new song with {output_length} characters.")
        if prompt is None:
            prompt = [0]

        self.model.eval()
        initial = torch.tensor([prompt]).long().to(get_device())  # assume 0 is start token

        sample = self.model.generate(initial, output_length, temperature=temperature, filter_thres=filter_treshold)
        return sample.cpu().detach().numpy()[0]

    def create_model(self):
        model = CompoundWordAutoregressiveWrapper(CompoundTransformerWrapper(
            num_tokens=self.num_tokens,
            max_seq_len=self.max_sequence_length,
            attn_layers=Decoder(
                dim=self.dim,
                depth=self.depth,
                heads=self.heads,
                attn_dropout=self.dropout,  # dropout post-attention
                ff_dropout=self.dropout,  # feedforward dropout
                rotary_pos_emb=True
            )
        ),
            ignore_index=0,
            pad_value=0
        ).to(get_device())

        return model

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'num_tokens': self.num_tokens,
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
