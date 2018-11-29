'''
Script that converts npy files into TFRecords files
'''
import sys
import os
sys.path.append(os.getcwd())

from Settings.arguments import arguments
from Training.tfrecords_converter import TFRecordsConverter

AVAILABLE_STREETS = [1,4]

error = Exception(''' Please specify the street.

	examples:
	python -m DataGeneration/main_data_generation.py --street 4
	python -m DataGeneration/main_data_generation.py --street=4

	available streets:
	1: preflop
	4: river
	''')

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

def street2name(street):
    if street == 1:
        return 'preflop'
    elif street == 2:
        return 'flop'
    elif street == 3:
        return 'turn'
    elif street == 4:
        return 'river'

args = sys.argv[1:]
street = parse_arguments(args)
street_name = street2name(street)


NPY_DIR_TRAIN = os.path.join(arguments.data_path, street_name, 'npy')
TFRECORDS_DIR_TRAIN = os.path.join(arguments.data_path, street_name, 'tfrecords')

def main():
    print('Initializing TFRecords Converter...')
    converter = TFRecordsConverter(arguments.tfrecords_batch_size)
    print('Converting NPY to TFRecords...')
    converter.convert_npy_to_tfrecords(NPY_DIR_TRAIN, TFRECORDS_DIR_TRAIN)
    print('Done!')




main()
