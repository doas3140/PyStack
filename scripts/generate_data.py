'''
	Script that generates ranges and cfvs.
'''
import sys
import os
os.chdir('..')
sys.path.append( os.path.join(os.getcwd(),'src') )

from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from DataGeneration.data_generation import DataGeneration

from arguments_parser import parse_arguments


def main():
	args = sys.argv[1:]
	street, starting_idx, approximate = parse_arguments(args)
	street_name = card_to_string.street2name(street)
	dirpath = os.path.join( arguments.data_path, street_name, '{}_{}'.format(approximate, 'npy') )
	data_generation = DataGeneration(dirpath)
	data_generation.generate_data(street, approximate, starting_idx)



main()
