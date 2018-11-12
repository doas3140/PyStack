'''
	Wraps the calls to the final neural net.
'''
import tensorflow.keras as keras
import tensorflow as tf

from Settings.arguments import arguments
from Nn.net_builder import nnBuilder
from Nn.basic_huber_loss import BasicHuberLoss

class ValueNn():
	def __init__(self):
		''' Loads the neural net from disk.
		'''
		# set input and output layer names
		# (must match keras model layer names)
		self.input_layer_name = 'input'
		self.output_layer_name = 'zero_sum_output'
		# set checkpoint and profiler (optional) dir
		self.model_dir = arguments.model_path
		self.profiler_dir = arguments.profiler_path
		# load keras model
		keras_model, self.x_shape, self.y_shape = nnBuilder.build_net()
		print('NN architecture:')
		keras_model.summary()
		# compile model
		self.compile_keras_model(keras_model)
		self.create_estimator(keras_model)


	def compile_keras_model(self, keras_model):
		print('Compiling model...')
		loss = BasicHuberLoss(delta=1.0)
		optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0)
		keras_model.compile(loss=loss, optimizer=optimizer)


	def create_estimator(self, keras_model):
		''' Creates estimator from keras_model
		@param: keras model object
		'''
		print('Creating Estimator...')
		config = tf.estimator.RunConfig(
			save_checkpoints_secs = 10*60, # save every 10 mins
			# save_checkpoints_steps = 100, # save every n steps
			keep_checkpoint_max = 10, # retain 10 recent checkpoints
			save_summary_steps = 100, # save summaries every n steps
		)
		self.estimator = tf.keras.estimator.model_to_estimator(
				keras_model = keras_model,
				keras_model_path = None,
				model_dir = self.model_dir,
				config = config )


	def get_value(self, inputs, output):
		''' Gives the neural net output for a batch of inputs.
		@param: inputs An (N,I) tensor containing N instances of
				neural net inputs. See @{net_builder} for details of each input.
		@param: output An (N,O) tensor in which to store N sets of
				neural net outputs. See @{net_builder} for details of each output.
		'''
		# create input function
		in_fn = tf.estimator.inputs.numpy_input_fn
		input_fn = in_fn( x={self.input_layer_name:some_x}, num_epochs=1, shuffle=False )
		# predict
		predictions = self.estimator.predict(input_fn=input_fn)
		# return predictions
		pred = [ p[self.output_layer_name] for p in predictions ]
		output[:,:] = np.array(pred)




#
