'''
	Script that trains the neural network.
	Uses data previously generated with @{data_generation_call}.
'''
import sys
import os
os.chdir('..')
sys.path.append( os.path.join(os.getcwd(),'src') )

import tensorflow as tf

from NnTraining.train import Train
from Game.card_to_string_conversion import card_to_string
from Settings.arguments import arguments


AVAILABLE_STREETS = [1,3,4]

error = Exception(''' Please specify the street.

	examples:
	python -m DataGeneration/main_data_generation.py --street 4
	python -m DataGeneration/main_data_generation.py --street=4

	available streets:
	1: preflop
	3: turn
	4: river
	''')



if arguments.XLA:
	config = tf.ConfigProto()
	config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
	sess = tf.Session(config=config)
	tf.keras.backend.set_session(sess)


def main():
	# parse CLI arguments
	args = sys.argv[1:]
	street = parse_arguments(args)
	street_name = card_to_string.street2name(street)
	# create data directories
	data_dirs = []
	data_dirs.append( os.path.join(os.getcwd(), 'Data', 'TrainSamples', street_name, 'tfrecords') )
	# data_dirs.append( os.path.join(arguments.data_path, street_name, 'tfrecords') )
	T = Train(data_dir_list=data_dirs, street=street)
	T.train(num_epochs=arguments.num_epochs, batch_size=arguments.batch_size)




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
