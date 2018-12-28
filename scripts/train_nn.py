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

from arguments_parser import parse_arguments


if arguments.XLA:
	config = tf.ConfigProto()
	config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
	sess = tf.Session(config=config)
	tf.keras.backend.set_session(sess)


def main():
	args = sys.argv[1:]
	street, starting_idx, approximate = parse_arguments(args)
	street_name = card_to_string.street_to_name(street)
	# create data directories
	data_dirs = []
	data_dirs.append( os.path.join(os.getcwd(), 'Data', 'TrainSamples', street_name, '{}_{}'.format(approximate, 'tfrecords')) )
	# data_dirs.append( os.path.join(r'D:\Datasets\Pystack\NoLimitTexasHoldem\river', 'tfrecords_1m_16') )
	T = Train(data_dir_list=data_dirs, street=street)
	T.train(num_epochs=arguments.num_epochs, batch_size=arguments.batch_size, validation_size=0.1, start_epoch=starting_idx)


main()
