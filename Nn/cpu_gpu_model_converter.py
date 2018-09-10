'''
	Generates a neural net model in CPU format from a neural net model saved
	in GPU format.
'''

from ..Settings.arguments import arguments

def convert_gpu_to_cpu(gpu_model_path):
	''' Generates a neural net model in CPU format from a neural net model saved
		in GPU format.
	@param: gpu_model_path the prefix of the path to the gpu model, which is
			appended with `_gpu.info` and `_gpu.model`
	'''
	pass


def convert_cpu_to_gpu(cpu_model_path):
	''' Generates a neural net model in GPU format from a neural net model saved
		in CPU format.
	@param: cpu_model_path the prefix of the path to the cpu model, which is
			appended with `_cpu.info` and `_cpu.model`
	'''
	pass




convert_gpu_to_cpu('../Data/Models/PotBet/final')
