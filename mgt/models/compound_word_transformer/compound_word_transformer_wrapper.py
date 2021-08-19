################################################################################
# Sampling
################################################################################
# -- temperature -- #
import numpy as np
import torch
from torch import nn
from x_transformers.x_transformers import AttentionLayers, default, AbsolutePositionalEmbedding, always

from mgt.models.compound_word_transformer.compound_transformer_embeddings import CompoundTransformerEmbeddings
from mgt.models.utils import get_device


def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().detach().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


class CompoundWordTransformerWrapper(nn.Module):
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

    def forward_output_sampling(self, h, y_type):
        # sample type
        y_type_logit = y_type[0, :]
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(np.array([cur_word_type])).long().to(get_device()).unsqueeze(0)

        tf_skip_type = self.word_emb_type(type_word_t)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        proj_barbeat = self.proj_barbeat(y_)
        proj_tempo = self.proj_tempo(y_)
        proj_instrument = self.proj_instrument(y_)
        proj_pitch = self.proj_pitch(y_)
        proj_duration = self.proj_duration(y_)
        proj_velocity = self.proj_velocity(y_)

        # sampling gen_cond
        cur_word_barbeat = sampling(proj_barbeat)
        cur_word_tempo = sampling(proj_tempo, p=0.9)
        cur_word_instrument = sampling(proj_instrument, p=0.9)
        cur_word_pitch = sampling(proj_pitch, p=0.9)
        cur_word_duration = sampling(proj_duration)
        cur_word_velocity = sampling(proj_velocity)

        # collect
        next_arr = np.array([
            cur_word_type,
            cur_word_barbeat,
            cur_word_tempo,
            cur_word_instrument,
            cur_word_pitch,
            cur_word_duration,
            cur_word_velocity
        ])
        return next_arr

    def forward_output(self,
                       h,
                       target
                       ):
        tf_skip_type = self.word_emb_type(target[..., 0])

        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        proj_barbeat = self.proj_barbeat(y_)
        proj_tempo = self.proj_tempo(y_)
        proj_instrument = self.proj_instrument(y_)
        proj_pitch = self.proj_pitch(y_)
        proj_duration = self.proj_duration(y_)
        proj_velocity = self.proj_velocity(y_)

        return proj_barbeat, proj_tempo, proj_instrument, proj_pitch, proj_duration, proj_velocity

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

        if not self.training:
            x.squeeze(0)

        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
        x = self.norm(x)

        return x, self.proj_type(x)
