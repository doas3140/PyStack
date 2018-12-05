''' Builds the neural net architecture.

	For M buckets, the neural net inputs have size 2*M+1, containing range
	vectors over buckets for each player, as well as a feature capturing the
	pot size. These are arranged as [{p1_range}, {p2_range}, pot_size].

	The neural net outputs have size 2*M, containing counterfactual value
	vectors over buckets for each player. These are arranged as
	[{p1_cfvs}, {p2_cfvs}].
'''

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
import tensorflow as tf

class NetBuilder():
	def __init__(self):
		pass


	def build_net(self):
		''' Builds a neural net with architecture specified by @{arguments.net}.
		@return a newly constructed neural net
		@return input shape (ex: [224,224,3] if img)
		@return output shape (ex: [10] if 10 classes)
		'''
		num_ranks, num_suits, num_cards = game_settings.rank_count, game_settings.suit_count, game_settings.card_count
		num_hands, num_players = game_settings.hand_count, constants.players_count
		# input and output parameters
		num_output = num_hands * num_players
		num_input = num_output + 1 + num_cards + num_suits + num_ranks
		input_shape = [num_input]
		output_shape = [num_output]
		# neural network architecture
		m_input = tf.keras.layers.Input(input_shape, name='input')
		# slicing off pot size and board ([1, hands x 2 + pot_size + board] -> [1, hands x 2])
		r = tf.keras.layers.Lambda(lambda x: x[ : , :num_output ], name='input_ranges')(m_input)
		# reconstruct mask from ranges
		mask = tf.keras.layers.Lambda(generate_mask, name='mask')(r)
		# feed forward part
		ff = m_input
		for i, num_neurons in enumerate(arguments.num_neurons):
			names = [s.format(i) for s in ('dense_{}', 'relu_{}', 'dropout_{}', 'batch_norm_{}')]
			ff = tf.keras.layers.Dense(num_neurons, name=names[0])(ff)
			# ff = tf.keras.layers.BatchNormalization(name=names[3])(ff)
			ff = tf.keras.layers.PReLU(name=names[1])(ff)
			# ff = tf.keras.layers.Dropout(rate=0.05, name=names[2])(ff)
		ff = tf.keras.layers.Dense(num_output, name='feed_forward_output')(ff)
		m_output = tf.keras.layers.multiply([ff,mask], name='masked_output')
		# # zero-sum output
		# # dot product of both (feed forward and player ranges)
		# d = tf.keras.layers.dot([ff,r], axes=1, name='dot_product')
		# # divide it by 2
		# d = tf.keras.layers.Lambda(lambda x: x/2, name='division_by_2')(d)
		# # subtract from neural net output
		# m_output = tf.keras.layers.subtract([ff,d], name='zero_sum_output')
		model = tf.keras.models.Model(m_input, m_output)
		return model, input_shape, output_shape


def generate_mask(ranges):
	zero = tf.constant(0.0, dtype=tf.float32)
	total_hands = tf.constant(1326*2, dtype=tf.float32)
	mask = tf.where( tf.greater(ranges, zero), tf.ones_like(ranges), tf.zeros_like(ranges) )
	return mask

nnBuilder = NetBuilder()
