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

AVAILABLE_STREETS = [1,3,4]

error = Exception(''' Please specify the street.

	examples:
	python -m DataGeneration/main_data_generation.py --street 4
	python -m DataGeneration/main_data_generation.py --street=4

	available streets:
	1: preflop
	3: turn
	4: river

	setting starting idx of filenames:
	python -m DataGeneration/main_data_generation.py --street 4 --start-idx 1
	''')


def main():
	args = sys.argv[1:]
	street, starting_idx = parse_arguments(args)
	street_name = card_to_string.street2name(street)

	dirpath = os.path.join(arguments.data_path, street_name, 'npy')
	data_generation = DataGeneration(dirpath)

	data_generation.generate_data(street, starting_idx)




def search_argument(name, args):
	for i, arg in enumerate(args):
		if name in arg:
			if '=' in arg:
				possible_result = arg.split('=')[-1]
			else:
				possible_result = args[i+1]
			try:
				return int(possible_result)
			except:
				raise(error)
	return None

def parse_arguments(args):
	street = search_argument('--street', args)
	idx = search_argument('--start-idx', args)
	if street is None or street not in AVAILABLE_STREETS:
		raise(error)
	if idx is None:
		idx = 0
	return street, idx

main()
