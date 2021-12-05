from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator

from mgt.models.transformer_model import TransformerModel

"""
Example showing how to save and load a model.
"""
dictionary = DictionaryGenerator.create_dictionary();
model = TransformerModel(dictionary)
model.save_checkpoint("test_model")
model2 = TransformerModel.load_checkpoint("test_model")
