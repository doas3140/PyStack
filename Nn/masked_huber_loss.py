'''
	Computes a Huber loss for neural net training and evaluation.
	Computes the loss across buckets, but only on buckets that are
	possible on a given board.
'''

from ..Settings.arguments import arguments

class MaskedHuberLoss():
	def __init__(self):
		self.criterion = nn.SmoothL1Criterion()


	def cuda(self):
		''' Moves the torch criterion (used for loss and gradient computation)
			to the GPU.
		@return the MaskedHuberLoss object that `cuda()` is called on
		'''
		self.criterion = self.criterion.cuda()
		return self


	def forward(self, outputs, targets, mask):
		''' Computes the loss over a batch of neural net outputs and targets.
		@param: outputs an (N,M) tensor containing N vectors of values over buckets,
				output by the neural net
		@param: targets an (N,M) tensor containing N vectors of actual values over
				buckets, produced by @{data_generation_call}
		@param: mask an (N,M) tensor containing N mask vectors generated with
		@{bucket_conversion.get_possible_bucket_mask}
		@return the sum of Huber loss applied elementwise on `outputs` and `targets`,
				masked so that only valid buckets are included
		'''
		pass
		# 1.0 zero out the outputs/target so that the error does not depend on these
		# 2.0 if the batch size has changed, create new storage for the sum,
		# otherwise reuse
		# 3.0 compute mask sum for each batch
		# 3.1 mask multiplier - note that mask is 1 for impossible features
		# 4.0 multiply to get a new losss
		# loss is not really computed batch-wise correctly,
    	# but that does not really matter now since gradients are correct



	def backward(self, outputs, targets, mask):
		''' Computes the gradient of the loss function @{forward} with
			arguments `outputs`, `targets`, and `mask`.
			Must be called after a @{forward} call with the same arguments.
		@param: outputs an (N,M) tensor containing N vectors of values over buckets,
				output by the neural net
		@param: targets an (N,M) tensor containing N vectors of actual values over
				buckets, produced by @{data_generation_call}
		@param: mask an (N,M) tensor containing N mask vectors generated with
				@{bucket_conversion.get_possible_bucket_mask}
		@return the gradient of @{forward} applied to the arguments
		'''
		pass
		# we use the multiplier computed with the mask during forward call




#
