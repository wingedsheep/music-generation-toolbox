from datetime import time

import numpy as np
import torch
import random
import time

from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

from encoders.dictionary import Dictionary


def create_batch(iterable, n=1):
    array_length = len(iterable)
    for ndx in range(0, array_length, n):
        yield iterable[ndx:min(ndx + n, array_length)]


def get_batches(x_train, batch_size):
    new_list = x_train.copy()
    random.shuffle(new_list)
    return list(create_batch(new_list, batch_size))


class ReformerModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 max_sequence_length=8192,
                 learning_rate=1e-3
                 ):
        self.dictionary = dictionary
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=0.1):
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")
            batches = get_batches(x_train, batch_size)
            epoch_losses = []
            for batch in batches:
                # when training, set return_loss equal to True
                batch = [x.long().cuda() for x in batch]
                loss = self.model(batch, return_loss=True)
                loss.backward()
                self.optimizer.step()
                loss_item = loss.item()
                epoch_losses.append(loss_item)
                print(f"Batch loss is {loss_item}.")

            epoch_loss = np.mean(epoch_losses)
            if epoch_loss <= stop_loss:
                print(f"Loss of {epoch_loss} was lower than stop loss of {stop_loss}. Stopping training.")
                return

            running_time = (time.time() - start_time)
            print(f"Loss after epoch {epoch + 1} is {epoch_loss}. Running time: {running_time}")

    def generate(self, output_length=100):
        self.model.eval()
        initial = torch.tensor([[0]]).long().cuda()  # assume 0 is start token
        sample = self.model.generate(initial, output_length, temperature=1., filter_thres=0.9)
        return sample.cpu().detach().numpy()[0]

    def create_model(self):
        model = ReformerLM(
            num_tokens=self.dictionary.size() + 1,
            dim=1024,
            depth=12,
            max_seq_len=self.max_sequence_length,
            lsh_dropout=0.1,
            causal=True,
            full_attn_thres=512
        ).cuda()

        # 0 is used for padding and no loss to be calculated on it
        return TrainingWrapper(model, ignore_index=0, pad_value=0)

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
