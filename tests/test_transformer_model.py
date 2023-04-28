import unittest

import numpy as np

from mgt.datamanagers.data_manager import Dictionary
from mgt.models.transformer_model import TransformerModel


class TestTransformerModel(unittest.TestCase):

    def test_learning(self):
        # Create a simple dictionary. 0 is the padding character.
        dictionary = Dictionary({str(i): i for i in range(10)}, {i: str(i) for i in range(10)})

        # Create a simple model with small dimensions and depth
        model = TransformerModel(
            dictionary=dictionary,
            max_sequence_length=15,
            learning_rate=1e-3,
            dropout=0.1,
            dim=128,
            depth=3,
            heads=3
        )

        # Create a simple song with a length of 50 tokens, with each token being a number between 1 and 9.
        simple_song = [str(i % 9 + 1) for i in range(50)]

        # Prepare the training data
        x_train = [np.array([dictionary.word_to_data(token) for token in simple_song])]

        # Train the model on the simple song
        model.train(x_train=x_train, epochs=50, batch_size=6, batches_per_epoch=10, report_per_x_batches=5)

        # Generate a song with the same length as the original one
        generated_song = model.generate(output_length=49, prompt=[dictionary.word_to_data('1')])

        # Convert the generated song back to tokens
        generated_song_tokens = [dictionary.data_to_word(token) for token in generated_song]
        generated_song_tokens.insert(0, '1')

        # Compare the generated song with the original one, there should be no more than 5 errors in subsequent tokens
        # A correct token is when a '1' is followed by a '2', a '2' is followed by a '3', etc.
        # and finally a '9' is followed by a '1'.
        errors = 0
        for i in range(1, len(generated_song_tokens)):
            expected_token = str(((int(generated_song_tokens[i - 1]) % 9) + 1))
            if generated_song_tokens[i] != expected_token:
                errors += 1

            if errors > 5:
                break

        self.assertTrue(errors <= 4, f"Generated song has more than 4 errors in subsequent tokens. Errors: {errors}")
        print(f"Generated song has {errors} errors in subsequent tokens.")


if __name__ == '__main__':
    unittest.main()
