from datetime import time

import torch
from torch import randint
import time

from reformer_pytorch import ReformerLM
from reformer_pytorch.generative_tools import TrainingWrapper

from encoders.dictionary import Dictionary


class ReformerModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 max_sequence_length=8192
                 ):
        self.dictionary = dictionary
        self.max_sequence_length = max_sequence_length
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, stop_loss=0.1):
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")
            # when training, set return_loss equal to True
            self.model.train()
            loss = self.model(x_train, return_loss=True)
            loss.backward()
            self.optimizer.step()
            loss_item = loss.item()
            if loss_item <= stop_loss:
                print(f"Loss of {loss_item} was lower than stop loss of {stop_loss}. Stopping training.")
                return
            running_time = (time.time() - start_time) / 1000
            print(f"Loss after epoch {epoch + 1} is {loss.item()}. Running time: {running_time}")

    def generate(self, output_length=100):
        initial = torch.tensor([[0]]).long()  # assume 0 is start token
        sample = self.model.generate(initial, output_length, temperature=1., filter_thres=0.9)
        return sample.cpu().detach().numpy()

    def create_model(self):
        model = ReformerLM(
            num_tokens=self.dictionary.size() + 1,
            dim=1024,
            depth=12,
            max_seq_len=self.max_sequence_length,
            lsh_dropout=0.1,
            causal=True,
            full_attn_thres=1024
        )

        # 0 is used for padding and no loss to be calculated on it
        return TrainingWrapper(model, ignore_index=0, pad_value=0)

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)
