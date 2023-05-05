import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mgt.models.compound_word_transformer.compound_word_transformer_utils import COMPOUND_WORD_PADDING, pad
from mgt.models.compound_word_transformer.compound_word_transformer_wrapper import CompoundWordTransformerWrapper
from mgt.models.utils import get_device


def type_mask(target):
    return target[..., 0] != 0


def calculate_loss(logits, targets, loss_mask):
    """
    Calculates the cross-entropy loss for the given logits and targets.
    :param logits: The logits with shape (batch_size, max_sequence_length - 1, classes)
    :param target: The targets with shape (batch_size, max_sequence_length - 1)
    :param loss_mask: The loss mask with shape (batch_size, max_sequence_length - 1)
    :return: The cross-entropy loss
    """

    trainable_values = torch.sum(loss_mask)
    if trainable_values.item() == 0:
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

    # Determine the number of classes from logits
    _, _, classes = logits.shape

    # Flatten logits and targets and mask using .reshape(...)
    logits_flattened = logits.reshape(-1, classes)      # Shape: (batch_size * (max_sequence_length - 1), classes)
    targets_flattened = targets.reshape(-1)             # Shape: (batch_size * (max_sequence_length - 1),)
    mask_flattened = loss_mask.reshape(-1)              # Shape: (batch_size * (max_sequence_length - 1),)

    # Compute the cross-entropy loss using F.cross_entropy
    loss = F.cross_entropy(logits_flattened, targets_flattened, reduction='none')

    # Apply the loss mask
    loss = loss * mask_flattened

    # Divide the loss by the number of trainable values
    loss = torch.sum(loss) / trainable_values

    return loss


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
    def generate(self, prompt, output_length=100, selection_temperatures=None, selection_probability_tresholds=None):
        self.net.eval()

        print('------ initiate ------')
        final_res = prompt.copy()
        last_token = final_res[-self.max_seq_len:]
        input_ = torch.tensor(np.array([last_token])).long().to(get_device())
        h, y_type = self.net.forward_hidden(input_)

        print('------ generate ------')
        for _ in range(output_length):
            # sample others
            next_arr = self.net.forward_output_sampling(
                h[:, -1:, :],
                y_type[:, -1:, :],
                selection_temperatures=selection_temperatures,
                selection_probability_tresholds=selection_probability_tresholds)

            final_res.append(next_arr.tolist())

            # forward
            last_token = final_res[-self.max_seq_len:]
            input_ = torch.tensor(np.array([last_token])).long().to(get_device())
            h, y_type = self.net.forward_hidden(input_)

        return final_res

    def train_step(self, x, **kwargs):
        xi = x[:, :-1, :]
        target = x[:, 1:, :]

        h, proj_type = self.net.forward_hidden(xi, **kwargs)
        proj_barbeat, proj_tempo, proj_instrument, proj_note_name, proj_octave, proj_duration, proj_velocity = self.net.forward_output(
            h, target)
        # Filter padding indices
        type_loss = calculate_loss(proj_type, target[..., 0], type_mask(target))
        barbeat_loss = calculate_loss(proj_barbeat, torch.clamp(target[..., 1], max=self.net.num_tokens[1] - 1), type_mask(target))
        tempo_loss = calculate_loss(proj_tempo, torch.clamp(target[..., 2], max=self.net.num_tokens[2] - 1), type_mask(target))
        instrument_loss = calculate_loss(proj_instrument, torch.clamp(target[..., 3], max=self.net.num_tokens[3] - 1), type_mask(target))
        note_name_loss = calculate_loss(proj_note_name, torch.clamp(target[..., 4], max=self.net.num_tokens[4] - 1), type_mask(target))
        octave_loss = calculate_loss(proj_octave, torch.clamp(target[..., 5], max=self.net.num_tokens[5] - 1), type_mask(target))
        duration_loss = calculate_loss(proj_duration, torch.clamp(target[..., 6], max=self.net.num_tokens[6] - 1), type_mask(target))
        velocity_loss = calculate_loss(proj_velocity, torch.clamp(target[..., 7], max=self.net.num_tokens[7] - 1), type_mask(target))

        return type_loss, barbeat_loss, tempo_loss, instrument_loss, note_name_loss, octave_loss, duration_loss, velocity_loss
