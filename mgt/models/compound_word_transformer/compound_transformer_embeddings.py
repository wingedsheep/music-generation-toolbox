import math

from torch import nn


class CompoundTransformerEmbeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(CompoundTransformerEmbeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

    def weight(self):
        return self.lut.weight
