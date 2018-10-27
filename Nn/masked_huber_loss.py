'''
	Computes a Huber loss for neural net training and evaluation.
	Computes the loss across buckets, but only on buckets that are
	possible on a given board.
'''

from ..Settings.arguments import arguments

class MaskedHuberLoss():
	def __init__(self):
		self.criterion = nn.SmoothL1Criterion()


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
		batch_size = outputs.shape[0]
		feature_size = outputs.shape[1]
		# 1.0 zero out the outputs/target so that the error does not depend on these
		outputs *= mask
  		targets *= mask
		loss = self.criterion.forward(outputs, targets)
		# 2.0 if the batch size has changed, create new storage for the sum, otherwise reuse
		if self.mask_sum is None or (self.mask_sum.shape[0] != batch_size):
			self.mask_placeholder = np.zeros(mask.shape, dtype=arguments.dtype)
			self.mask_sum = np.zeros([batch_size], dtype=arguments.dtype)
			self.mask_multiplier = self.mask_sum.copy().reshape([-1,1])
		# 3.0 compute mask sum for each batch
		self.mask_placeholder = mask.copy()
		self.mask_sum = np.sum(self.mask_placeholder, axis=1)
		# 3.1 mask multiplier - note that mask is 1 for impossible features
		self.mask_multiplier.fill(feature_size)
		self.mask_multiplier -= self.mask_sum
		self.mask_multiplier /= feature_size
		# 4.0 multiply to get a new loss
		# loss is not really computed batch-wise correctly,
    	# but that does not really matter now since gradients are correct
		loss_multiplier = (batch_size * feature_size) / (batch_size * feature_size - self.mask_sum.sum() )
		new_loss = loss_multiplier * loss



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
		dloss_doutput = self.criterion.backward(outputs, targets)
		# we use the multiplier computed with the mask during forward call
		dloss_doutput /= self.mask_multiplier:expandAs(dloss_doutput)
		return dloss_doutput




#
