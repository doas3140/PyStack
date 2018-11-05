'''
	Generates neural net training data by evaluating terminal equity for poker situations.

	Evaluates terminal equity (assuming both players check/call to the end of the game)
	instead of re-solving. Used for debugging.
'''

from Settings.arguments import arguments
from Settings.constants import constants
from Settings.game_settings import game_settings
from DataGeneration.range_generator import RangeGenerator
from Nn.bucketer import Bucketer
from Nn.bucket_conversion import BucketConversion
from TerminalEquity.terminal_equity import TerminalEquity

class DataGenerationCall():
	def __init__(self):
		pass

	def generate_data(self, train_data_count, valid_data_count):
		''' Generates training and validation files by evaluating terminal
			equity for random poker situations.
			Makes two calls to @{generate_data_file}. The files are saved to
			@{arguments.data_path}, respectively appended with `valid` and `train`.
		@param: train_data_count the number of training examples to generate
		@param: valid_data_count the number of validation examples to generate
		'''
		# valid data generation
		local file_name = arguments.data_path + 'valid'
		self.generate_data_file(valid_data_count, file_name)
		# train data generation
		file_name = arguments.data_path + 'train'
		self.generate_data_file(train_data_count, file_name)


	def generate_data_file(self, data_count, file_name):
		''' Generates data files containing examples of random poker situations
			with associated terminal equity.
			Each poker situation is randomly generated using @{range_generator} and
			@{random_card_generator}.
			For description of neural net input and target type, see @{net_builder}.
		@param: data_count the number of examples to generate
		@param: file_name the prefix of the files where the data is saved (appended
				with `.inputs`, `.targets`, and `.mask`).
		'''
		range_generator = RangeGenerator()
		batch_size = arguments.gen_batch_size
		assert(data_count % batch_size == 0, 'data count has to be divisible by the batch size')
		batch_count = data_count / batch_size
		bucketer = Bucketer()
		bucket_count = bucketer.get_bucket_count()
		player_count = 2
		target_size = bucket_count * player_count
		targets = np.zeros([data_count, target_size], dtype=arguments.dtype)
		input_size = bucket_count * player_count + 1
		inputs = np.zeros([data_count, input_size], dtype=arguments.dtype)
		mask = np.zeros([data_count, bucket_count], dtype=arguments.dtype)
		bucket_conversion = BucketConversion()
		equity = TerminalEquity()
		for batch in range(1, batch_count+1):
			board = card_generator.generate_cards(game_settings.board_card_count)
			range_generator.set_board(board)
			bucket_conversion.set_board(board)
			equity.set_board(board)
			# generating ranges
			ranges = np.zeros([player_count, batch_size, game_settings.card_count], dtype=arguments.dtype)
			for player in range(1, player_count+1):
				range_generator.generate_range(ranges[player])
			pot_sizes = np.zeros([arguments.gen_batch_size, 1], dtype=arguments.dtype)
			# generating pot features
			pot_sizes = np.random.rand(batch_size)
			# translating ranges to features
			batch_index = ( (batch-1)*batch_size+1, batch*batch_size )
			b_start, b_end = batch_index
			pot_feature_index =  -1
			inputs[ b_start:b_end , pot_feature_index ] = pot_sizes.copy()
			player_indexes = [(1,bucket_count), (bucket_count+1,bucket_count*2)]
			for player in range(1, player_count+1):
				p_start, p_end = player_indexes[player]
				bucket_conversion:card_range_to_bucket_range(ranges[player], inputs[ b_start:b_end , p_start:p_end ])
			# computaton of values using terminal equity
			values = np.zeros([player_count, batch_size, game_settings.card_count], dtype=arguments.dtype)
			for player in range(1, player_count+1):
				opponent = 3 - player
				equity.call_value(ranges[opponent], values[player])
			# translating values to nn targets
			for player in range(1, player_count+1):
				p_start, p_end = player_indexes[player]
				bucket_conversio.card_range_to_bucket_range(values[player], targets[ b_start:b_end , p_start:p_end ])
			# computing a mask of possible buckets
			bucket_mask = bucket_conversion.get_possible_bucket_mask()
			mask[ b_start:b_end , : ] = bucket_mask.copy() * np.ones([batch_size, bucket_count], dtype=arguments.dtype)
		np.save(file_name + '.inputs', inputs)
		np.save(file_name + '.targets', targets)
		np.save(file_name + '.mask', mask)




data_generation_call = DataGenerationCall()
