'''
	Script that trains the neural network.
	Uses data previously generated with @{data_generation_call}.
'''
import sys
import os
sys.path.append(os.getcwd())

from Nn.net_builder import nnBuilder
from Training.train import Train
from Settings.arguments import arguments

def main():
	train_dir = os.path.join(arguments.data_path, 'tfrecords')
	T = Train(train_dir=train_dir)
	T.train(steps=10000)




main()
