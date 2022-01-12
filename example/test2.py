import torch
from pretty_midi import pretty_midi, np

from example.vq_vae import VQVAE
from mgt.datamanagers.encoded_beats.beat_data_extractor2 import BeatDataExtractor2

data_extractor = BeatDataExtractor2()

# Prepare the dataset
midi_data = pretty_midi.PrettyMIDI("/Users/vincentbons/Documents/Music toolbox/simplified midi set/Red hot chili peppers/californication.mid")
data = data_extractor.extract_beats(midi_data)

# print(data)
# print(data_extractor.create_matrix_representation(data[10]))

encoded = [data_extractor.create_matrix_representation(x) for x in data]

# Create a VQ-VAE model and train it on the encoded bars.
# The model should learn to generate bars for all instruments.
model = VQVAE(in_channel=4)

criterion = torch.nn.MSELoss()

latent_loss_weight = 0.25

mse_sum = 0
mse_n = 0

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

training_data = torch.tensor(encoded).float()
sample_size = len(training_data)

model.train()
for i in range(10):
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
print(model.forward(training_data)[0])
model_output = np.abs(np.rint(model.forward(training_data[0:10])[0].detach().numpy())).astype(int)
