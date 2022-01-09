import torch

from example.vq_vae import VQVAE
from mgt.datamanagers.remi_data_manager import RemiDataManager
import numpy as np


def pad_right(data, padding_character=0, padding_length=512):
    while len(data) < padding_length:
        data.append(padding_character)
    return data


def split_into_bars(data, split_word=2, padding_character=0, padding_length=256):
    result = []
    split = []
    for index, value in enumerate(data):
        if value == split_word:
            if len(split) > 0:
                result.append(pad_right(split, padding_character, padding_length))
                split = []
        else:
            split.append(value)
    if len(split) > 0:
        result.append(pad_right(split, padding_character, padding_length))

    return result


def encode(data, max_bits):
    return [encode_word(x, max_bits) for x in data]


def encode_word(word, max_bits):
    binary = bin(word).replace("0b", "").zfill(max_bits)
    return [int(x) for x in binary]


def decode_word(binary_word):
    return int(''.join([str(x) for x in binary_word]), 2)


def decode_bar(binary_bar):
    return [decode_word(x) for x in binary_bar]


def decode(encoded_data):
    decoded_bars = [decode_bar(x) for x in encoded_data]
    words = []
    bar_character = 2
    for bar in decoded_bars:
        words.append(bar_character)
        for word in bar:
            if word != 0:
                words.append(word)
            else:
                break
    return words


def decode_model_output(model_output):
    encoded_data = [x[0] for x in model_output]
    return decode(encoded_data)


def encode_one_hot(data, number_of_tokens):
    return [encode_word_one_hot(x, number_of_tokens) for x in data]


def encode_word_one_hot(word, number_of_tokens):
    one_hot = np.zeros(number_of_tokens)
    one_hot[word] = 1
    return one_hot


def decode_word_one_hot(one_hot):
    return np.where(one_hot == 1)[0][0]


def decode_bar_one_hot(one_hot_bar):
    return [decode_word_one_hot(x) for x in one_hot_bar]


def decode_one_hot(encoded_data):
    decoded_bars = [decode_bar_one_hot(x) for x in encoded_data]
    words = []
    bar_character = 2
    for bar in decoded_bars:
        words.append(bar_character)
        for word in bar:
            if word != 0:
                words.append(word)
            else:
                break
    return words


def decode_model_output_one_hot(model_output):
    encoded_data = [x[0] for x in model_output]
    return decode_one_hot(encoded_data)


midis = ["/Users/vincentbons/Git/Vincent/music-generation-toolbox/data/pop/001.mid"]

# Parse MIDI as REMI
datamanager = RemiDataManager()
dataset = datamanager.prepare_data(midis)

# Encode the words as binary and split them into bars
number_of_words = 512
max_bits_for_binary_word_encoding = 16

bars = split_into_bars(dataset.data[0])
encoded_bars = [encode(bar, max_bits_for_binary_word_encoding) for bar in bars]

print(decode(encoded_bars))

# Create a VQ-VAE model and train it on the encoded bars.
# The model should learn to generate bars for all instruments.
model = VQVAE(in_channel=1)

criterion = torch.nn.MSELoss()

latent_loss_weight = 0.25

mse_sum = 0
mse_n = 0

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

training_data = torch.tensor([[bar] for bar in encoded_bars]).float()
sample_size = len(training_data)

model.train()
for i in range(200):
    model.zero_grad()

    batch = training_data

    out, latent_loss = model(batch)
    recon_loss = criterion(out, batch)
    latent_loss = latent_loss.mean()
    loss = recon_loss + latent_loss_weight * latent_loss
    loss.backward()

    optimizer.step()

    print(f"Epoch {i}: {loss}")

# Reconstruct from model output
model.eval()
print(model.forward(training_data[0:10])[0])
model_output = np.abs(np.rint(model.forward(training_data[0:10])[0].detach().numpy())).astype(int)
decoded_reconstruction = decode_model_output(model_output)

# Write to midi
remi_midi = datamanager.to_midi(decoded_reconstruction)
remi_midi.save("remi.midi")
