from mgt.datamanagers.data_helper import DataHelper
from mgt.datamanagers.remi.dictionary_generator import DictionaryGenerator

dataset = DataHelper.load('/Users/vincentbons/Documents/Music toolbox/lakh_remi_0')
dictionary = DictionaryGenerator.create_dictionary()

bar_character = str(dictionary.word_to_data('Bar_None'))

text = ''
for line in dataset:
    song = ' ' + ' '.join([str(x) for x in line]) + ' '
    split_text = song.split(' ' + bar_character + ' ')
    for t in split_text:
        text += t + '\n'

with open('result.txt', 'w') as f:
    f.write(text)
