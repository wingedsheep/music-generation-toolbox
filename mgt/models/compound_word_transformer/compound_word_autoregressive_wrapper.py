import random

import torch
from torch import nn
from x_transformers.x_transformers import default

from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_PADDING, pad
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device
import torch.nn.functional as F


def truncate_sequence(inputs, mask=None, pad_value=COMPOUND_WORD_PADDING):
    ba, t, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    mask = default(mask, torch.ones_like(inputs).bool())
    rand_length = random.randint(2, t)
    return inputs[:, :rand_length], mask[:, :rand_length]


class CompoundWordAutoregressiveWrapper(nn.Module):
    def __init__(self, net: CompoundWordTransformerWrapper, ignore_index=-100, pad_value=None):
        super().__init__()
        if pad_value is None:
            pad_value = COMPOUND_WORD_PADDING
        self.pad_value = pad_value
        self.ignore_index = ignore_index
        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, prompt, output_length=100):
        self.net.eval()

        print('------ initiate ------')
        final_res = prompt.copy()
        padded = pad(final_res[-self.max_seq_len:], self.max_seq_len)
        input_ = torch.tensor([padded]).long().to(get_device())
        h, y_type = self.net.forward_hidden(input_)

        print('------ generate ------')
        for _ in range(output_length):
            # sample others
            next_arr = self.net.forward_output_sampling(h[:, -1:, :], y_type[:, -1:, :])
            final_res.append(next_arr.tolist())

            # forward
            padded = pad(final_res[-self.max_seq_len:], self.max_seq_len)
            input_ = torch.tensor([padded]).long().to(get_device())

            h, y_type = self.net.forward_hidden(input_)

        return final_res

    def calculate_loss(self, predicted, target, loss_mask):
        loss = F.cross_entropy(predicted[:, ...].permute(0, 2, 1), target, reduction='none')
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def train_step(self, x, **kwargs):
        xi = x[:, :-1]
        target = x[:, 1:]

        mask = (target[..., 0] != 0)

        h, proj_type = self.net.forward_hidden(xi, **kwargs)
        proj_barbeat, proj_tempo, proj_instrument, proj_pitch, proj_duration, proj_velocity = self.net.forward_output(h, target)

        # Filter padding indices
        type_loss = self.calculate_loss(proj_type, target[..., 0], mask)
        barbeat_loss = self.calculate_loss(proj_barbeat, target[..., 1], mask)
        tempo_loss = self.calculate_loss(proj_tempo, target[..., 2], mask)
        instrument_loss = self.calculate_loss(proj_instrument, target[..., 3], mask)
        pitch_loss = self.calculate_loss(proj_pitch, target[..., 4], mask)
        duration_loss = self.calculate_loss(proj_duration, target[..., 5], mask)
        velocity_loss = self.calculate_loss(proj_velocity, target[..., 6], mask)

        return type_loss, barbeat_loss, tempo_loss, instrument_loss, pitch_loss, duration_loss, velocity_loss