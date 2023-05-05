import unittest

from mgt.models.compound_word_transformer_model import CompoundWordTransformerModel


def calc_first_index_error(compound_word, prev_compound_word):
    if compound_word[0] == 1:
        if prev_compound_word[0] != 3:
            return 1
    elif compound_word[0] != prev_compound_word[0] + 1:
        return 1
    return 0


def calc_other_indices_error(compound_word, prev_compound_word):
    errors = 0
    for i in range(1, len(compound_word)):
        if compound_word[i] == 1:
            if prev_compound_word[i] != 8:
                errors += 1
        elif compound_word[i] != prev_compound_word[i] + 1:
            errors += 1
    return errors


class TestCompoundWordTransformerModel(unittest.TestCase):

    def test_learning(self):
        # Create a simple model with small dimensions and depth
        model = CompoundWordTransformerModel(
            max_sequence_length=25,
            learning_rate=1e-3,
            dropout=0.1,
            dim=128,
            depth=3,
            heads=3
        )

        # Create a simple song of 60 compound words, each consisting of 8 properties
        simple_song = [
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 1],
                [3, 4, 5, 6, 7, 8, 1, 2],
                [1, 5, 6, 7, 8, 1, 2, 3],
                [2, 6, 7, 8, 1, 2, 3, 4],
                [3, 7, 8, 1, 2, 3, 4, 5],
                [1, 8, 1, 2, 3, 4, 5, 6],
                [2, 1, 2, 3, 4, 5, 6, 7],
                [3, 2, 3, 4, 5, 6, 7, 8],
                [1, 3, 4, 5, 6, 7, 8, 1],
                [2, 4, 5, 6, 7, 8, 1, 2],
                [3, 5, 6, 7, 8, 1, 2, 3],
                [1, 6, 7, 8, 1, 2, 3, 4],
                [2, 7, 8, 1, 2, 3, 4, 5],
                [3, 8, 1, 2, 3, 4, 5, 6],
                [1, 1, 2, 3, 4, 5, 6, 7],
                [2, 2, 3, 4, 5, 6, 7, 8],
                [3, 3, 4, 5, 6, 7, 8, 1],
                [1, 4, 5, 6, 7, 8, 1, 2],
                [2, 5, 6, 7, 8, 1, 2, 3],
                [3, 6, 7, 8, 1, 2, 3, 4],
                [1, 7, 8, 1, 2, 3, 4, 5],
                [2, 8, 1, 2, 3, 4, 5, 6],
                [3, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 1],
                [3, 4, 5, 6, 7, 8, 1, 2],
                [1, 5, 6, 7, 8, 1, 2, 3],
                [2, 6, 7, 8, 1, 2, 3, 4],
                [3, 7, 8, 1, 2, 3, 4, 5],
                [1, 8, 1, 2, 3, 4, 5, 6],
                [2, 1, 2, 3, 4, 5, 6, 7],
                [3, 2, 3, 4, 5, 6, 7, 8],
                [1, 3, 4, 5, 6, 7, 8, 1],
                [2, 4, 5, 6, 7, 8, 1, 2],
                [3, 5, 6, 7, 8, 1, 2, 3],
                [1, 6, 7, 8, 1, 2, 3, 4],
                [2, 7, 8, 1, 2, 3, 4, 5],
                [3, 8, 1, 2, 3, 4, 5, 6],
                [1, 1, 2, 3, 4, 5, 6, 7],
                [2, 2, 3, 4, 5, 6, 7, 8],
                [3, 3, 4, 5, 6, 7, 8, 1],
                [1, 4, 5, 6, 7, 8, 1, 2],
                [2, 5, 6, 7, 8, 1, 2, 3],
                [3, 6, 7, 8, 1, 2, 3, 4],
                [1, 7, 8, 1, 2, 3, 4, 5],
                [2, 8, 1, 2, 3, 4, 5, 6],
                [3, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 1],
                [3, 4, 5, 6, 7, 8, 1, 2],
                [1, 5, 6, 7, 8, 1, 2, 3],
                [2, 6, 7, 8, 1, 2, 3, 4],
                [3, 7, 8, 1, 2, 3, 4, 5],
                [1, 8, 1, 2, 3, 4, 5, 6],
                [2, 1, 2, 3, 4, 5, 6, 7],
                [3, 2, 3, 4, 5, 6, 7, 8],
                [1, 3, 4, 5, 6, 7, 8, 1],
                [2, 4, 5, 6, 7, 8, 1, 2],
                [3, 5, 6, 7, 8, 1, 2, 3],
                [1, 6, 7, 8, 1, 2, 3, 4]
            ]
        ]

        # Train the model on the simple song
        model.train(x_train=simple_song, epochs=50, batch_size=6, batches_per_epoch=10, report_per_x_batches=5)

        # Generate a song with the same length as the original one
        only_first_compound_word = simple_song[0][0:1]
        generated_song = model.generate(output_length=30, prompt=only_first_compound_word)

        # Print the first 10 words of the generated song
        print(generated_song[0:10])

        # Compare the generated song with the original one, there should be no more than 5 errors in subsequent tokens
        # [1, 2, 3, 4, 5, 6, 7, 8] should be followed by [2, 3, 4, 5, 6, 7, 8, 1] and so on.
        # Count errors as the number of tokens that are not incremented by 1, modulo 8 and module 3 for the first token.
        errors = 0
        error_list = []
        for i in range(1, len(generated_song)):
            errors_for_index = 0
            first_index_error = calc_first_index_error(generated_song[i], generated_song[i - 1])
            other_indices_error = calc_other_indices_error(generated_song[i], generated_song[i - 1])
            errors += first_index_error + other_indices_error
            errors_for_index += first_index_error + other_indices_error
            error_list.append(errors_for_index)

            if errors >= 5:
                break

        print(error_list)

        self.assertTrue(errors <= 5, f"Too many errors in the generated song: {errors}")
        print(f"Generated song has {errors} errors in subsequent tokens.")


if __name__ == '__main__':
    unittest.main()
