from mgt.datamanagers.data_manager import Dictionary
from mgt.datamanagers.remi_data_manager import RemiDataManager
from mgt.models.reformer_model import ReformerModel

import os
import glob


def run():
    """
    Example showing how we can train a new model and generate new music with it.
    """

    dict = [{}, {}]

    def append_to_dictionary(word):
        if word not in dict[0]:
            offset = len(dict[0])
            dict[0].update({word: offset})
            dict[1].update({offset: word})

    append_to_dictionary("pad")
    append_to_dictionary("mask")

    for i in range(2000):
        append_to_dictionary(str(i))

    dictionary = Dictionary(dict[0], dict[1])

    model = ReformerModel(dictionary, max_sequence_length=512, full_attn_thres=256)

    data = [
        [i for i in range(600)],
        [i for i in reversed(range(600))]
    ]

    print(data)

    print("Created model. Starting training for 50 epochs.")
    model.train(x_train=data, epochs=50, stop_loss=0.1)

    # Generate music
    print("Generating music.")
    output = model.generate(600)
    print(output)


run()