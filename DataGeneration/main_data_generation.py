'''
	Script that generates training and validation files.
'''
import sys
import os
sys.path.append(os.getcwd())

from Settings.arguments import arguments
from DataGeneration.data_generation import data_generation

data_generation.generate_data(arguments.gen_data_count, arguments.gen_num_files)
