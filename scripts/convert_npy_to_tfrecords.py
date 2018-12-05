'''
Script that converts npy files into TFRecords files
'''
import sys
import os
os.chdir('..')
sys.path.append( os.path.join(os.getcwd(),'src') )

from Settings.arguments import arguments
from Game.card_to_string_conversion import card_to_string
from NnTraining.tfrecords_converter import TFRecordsConverter

AVAILABLE_STREETS = [1,4]

error = Exception(''' Please specify the street.

	examples:
	python -m DataGeneration/main_data_generation.py --street 4
	python -m DataGeneration/main_data_generation.py --street=4

	available streets:
	1: preflop
	4: river
	''')



def main():
	# parse CLI arguments
	args = sys.argv[1:]
	street = parse_arguments(args)
	street_name = card_to_string.street2name(street)
	# directories
	NPY_DIR_TRAIN = os.path.join(arguments.data_path, street_name, 'npy')
	TFRECORDS_DIR_TRAIN = os.path.join(arguments.data_path, street_name, 'tfrecords')
	print('Initializing TFRecords Converter...')
	converter = TFRecordsConverter(arguments.tfrecords_batch_size)
	print('Converting NPY to TFRecords...')
	converter.convert_npy_to_tfrecords(NPY_DIR_TRAIN, TFRECORDS_DIR_TRAIN)
	print('Done!')




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
	if street is None or street not in AVAILABLE_STREETS:
		raise(error)
	return street

main()
