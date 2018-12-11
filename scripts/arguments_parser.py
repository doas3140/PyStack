

AVAILABLE_STREETS = [1,2,3,4]

AVAILABLE_APPROXIMATIONS = ['root_nodes', 'leaf_nodes']


error = Exception(''' Please specify the street.

	examples:
	python -m DataGeneration/main_data_generation.py --street 4
	python -m DataGeneration/main_data_generation.py --street=4

	available streets:
	1: preflop
	2: flop
	3: turn
	4: river

	setting to approximate root nodes or leaf nodes:
	python -m DataGeneration/main_data_generation.py --street 4 --approximate root_nodes
	python -m DataGeneration/main_data_generation.py --street 4 --approximate leaf_nodes
	(if none defined, then root_nodes is used)

	setting starting idx of filenames:
	python -m DataGeneration/main_data_generation.py --street 4 --approximate root_nodes --start-idx 1
	(if none defined, then 0 is used)
	''')


def search_argument(name, args, string=False):
	for i, arg in enumerate(args):
		if name in arg:
			if '=' in arg:
				possible_result = arg.split('=')[-1]
			else:
				possible_result = args[i+1]
			if string: return possible_result
			try:
				return int(possible_result)
			except:
				raise(error)
	return None

def parse_arguments(args):
	street = search_argument('--street', args)
	idx = search_argument('--start-idx', args)
	approximate = search_argument('--approximate', args, string=True)
	if street is None or street not in AVAILABLE_STREETS:
		raise(error)
	if idx is None:
		idx = 0
	if approximate is None:
		approximate = 'root_nodes'
	if approximate not in AVAILABLE_APPROXIMATIONS:
		raise(error)
	return street, idx, approximate
