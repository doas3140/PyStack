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

from arguments_parser import parse_arguments



def main():
	# parse CLI arguments
	args = sys.argv[1:]
	street, starting_idx, approximate = parse_arguments(args)
	street_name = card_to_string.street_to_name(street)
	# directories
	NPY_DIR = os.path.join( arguments.data_path, street_name, '{}_{}'.format(approximate, 'npy') )
	TFRECORDS_DIR = os.path.join( arguments.data_path, street_name, '{}_{}'.format(approximate, 'tfrecords') )
	print('Initializing TFRecords Converter...')
	converter = TFRecordsConverter(arguments.tfrecords_batch_size)
	print('Converting NPY to TFRecords...')
	converter.convert_npy_to_tfrecords(NPY_DIR, TFRECORDS_DIR, starting_idx)
	print('Done!')




main()
