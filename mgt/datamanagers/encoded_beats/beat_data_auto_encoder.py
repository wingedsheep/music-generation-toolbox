import torch
from sequitur.models import LINEAR_AE
import numpy as np
from torch import nn

from mgt.models import utils

defaults = {
    'input_dim': 2048,
    'encoding_dim': 512,
    'hidden_dims': [1024, 1024, 512, 512, 512],
    'hidden_activation': nn.ReLU(),
    'encoder_output_activation': nn.ReLU()
}


def get_batch(data, batch_size):
    indices = np.random.choice(len(data), size=batch_size)
    return torch.tensor(data[indices]).float().to(utils.get_device())


class BeatDataAutoEncoder(object):

    def __init__(self,
                 input_dim: int = defaults['input_dim'],
                 encoding_dim: int = defaults['encoding_dim'],
                 hidden_dims: [int] = defaults['hidden_dims'],
                 hidden_activation=defaults['hidden_activation'],
                 encoder_output_activation=defaults['encoder_output_activation']
                 ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.encoder_output_activation = encoder_output_activation
        self.model = self.create_model()

    def train(self, training_data, nr_of_batches: int, batch_size: int = 256, learning_rate: float = 1e-4):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(nr_of_batches):
            optimizer.zero_grad()

            # Forward pass
            batch = get_batch(training_data, batch_size)
            x_prime = self.model(batch)

            loss = criterion(x_prime, batch)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            optimizer.step()

            print(f"Epoch {epoch}, loss {loss.item()}")

    def encode(self, data):
        tensor = torch.tensor(data).float().to(utils.get_device())
        return self.model.encoder.forward(tensor).cpu().detach().numpy()

    def decode(self, encoded):
        tensor = torch.tensor(encoded).float().to(utils.get_device())
        return [round(x) for x in self.model.decoder.forward(tensor).cpu().detach().numpy()]

    def create_model(self):
        return LINEAR_AE(
            input_dim=self.input_dim,
            encoding_dim=self.encoding_dim,
            h_dims=self.hidden_dims,
            h_activ=self.hidden_activation,
            out_activ=self.encoder_output_activation
        ).to(utils.get_device())
