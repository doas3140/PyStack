'''
	Script that generates training and validation files.
'''

from Settings.arguments import arguments
from DataGeneration.data_generation import data_generation

data_generation.generate_data(arguments.train_data_count, arguments.valid_data_count)
