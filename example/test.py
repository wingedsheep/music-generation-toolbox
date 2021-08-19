from mgt.datamanagers.compound_word_data_manager import CompoundWordDataManager

data_manager = CompoundWordDataManager()
remi = data_manager.to_remi([[3, 0, 0, 0, 0, 0, 0]])
print(remi)
