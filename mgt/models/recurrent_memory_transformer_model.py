import time

import torch
import numpy as np
from recurrent_memory_transformer_pytorch import RecurrentMemoryTransformer, RecurrentMemoryTransformerWrapper
from mgt.datamanagers.data_manager import Dictionary
from mgt.models import utils

defaults = {
    'learning_rate': 1e-4,
    'dropout': 0.1,
    'dim': 512,
    'depth': 6,
    'heads': 8,
    'num_memory_tokens': 128,
    'seq_len': 1024,
    'use_flash_attn': True,
    'use_xl_memories': True,
    'xl_mem_len': 512
}


class RecurrentMemoryTransformerModel(object):

    def __init__(self,
                 dictionary: Dictionary,
                 learning_rate=defaults['learning_rate'],
                 dropout=defaults['dropout'],
                 dim=defaults['dim'],
                 depth=defaults['depth'],
                 heads=defaults['heads'],
                 num_memory_tokens=defaults['num_memory_tokens'],
                 seq_len=defaults['seq_len'],
                 use_flash_attn=defaults['use_flash_attn'],
                 use_xl_memories=defaults['use_xl_memories'],
                 xl_mem_len=defaults['xl_mem_len']
                 ):
        self.dictionary = dictionary
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.num_memory_tokens = num_memory_tokens
        self.seq_len = seq_len
        self.use_flash_attn = use_flash_attn
        self.use_xl_memories = use_xl_memories
        self.xl_mem_len = xl_mem_len
        self.model = self.create_model()
        self.optimizer = self.create_optimizer()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        self.optimizer = self.create_optimizer()

    def train(self, x_train, epochs, batch_size=4, stop_loss=None, batches_per_epoch=100, report_per_x_batches=20,
              gradient_accumulation_steps=1, num_segments=8):
        sequence_length_including_memory = num_segments * self.seq_len
        self.model.train()
        start_time = time.time()
        for epoch in range(epochs):
            print(f"Training epoch {epoch + 1}.")

            epoch_losses = []
            batch_losses = []
            nr_of_batches_processed = 0
            for _ in range(batches_per_epoch):

                for _ in range(gradient_accumulation_steps):
                    batch = utils.get_batch(
                        x_train,
                        batch_size=batch_size,
                        max_sequence_length=sequence_length_including_memory)

                    torch_batch = torch.tensor(np.array(batch)).long().to(utils.get_device())

                    loss = self.model(torch_batch, memory_replay_backprop=True)

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

        if not isinstance(self.model, RecurrentMemoryTransformerWrapper):
            raise ValueError("generate method requires a RecurrentMemoryTransformerWrapper instance")

        self.model.eval()
        initial = torch.tensor([prompt]).long().to(utils.get_device())  # assume 0 is start token

        generated = self.model.generate(initial, length=output_length, temperature=temperature, filter_thres=filter_treshold)

        return np.array(generated)

    def create_model(self):
        model = RecurrentMemoryTransformer(
            num_tokens=self.dictionary.size(),
            num_memory_tokens=self.num_memory_tokens,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            seq_len=self.seq_len,
            use_flash_attn=self.use_flash_attn,
            causal=True,
            dim_head=64
        )

        if self.use_xl_memories:
            model = RecurrentMemoryTransformerWrapper(model)

        return model.to(utils.get_device())

    def create_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'dictionary': self.dictionary,
            'learning_rate': self.learning_rate,
            'dropout': self.dropout,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'num_memory_tokens': self.num_memory_tokens,
            'seq_len': self.seq_len,
            'use_flash_attn': self.use_flash_attn,
            'use_xl_memories': self.use_xl_memories,
            'xl_mem_len': self.xl_mem_len,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    @staticmethod
    def load_checkpoint(path):
        checkpoint = torch.load(path)
        model = RecurrentMemoryTransformerModel(
            dictionary=checkpoint['dictionary'],
            learning_rate=utils.get_or_default(checkpoint, 'learning_rate', defaults),
            dropout=utils.get_or_default(checkpoint, 'dropout', defaults),
            dim=utils.get_or_default(checkpoint, 'dim', defaults),
            depth=utils.get_or_default(checkpoint, 'depth', defaults),
            heads=utils.get_or_default(checkpoint, 'heads', defaults),
            num_memory_tokens=utils.get_or_default(checkpoint, 'num_memory_tokens', defaults),
            seq_len=utils.get_or_default(checkpoint, 'seq_len', defaults),
            use_flash_attn=utils.get_or_default(checkpoint, 'use_flash_attn', defaults),
            use_xl_memories=utils.get_or_default(checkpoint, 'use_xl_memories', defaults),
            xl_mem_len=utils.get_or_default(checkpoint, 'xl_mem_len', defaults)
        )

        model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        return model
