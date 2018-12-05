
import sys
import os
sys.path.append(os.getcwd())

import tensorflow as tf

from Settings.arguments import arguments
from Nn.net_builder import nnBuilder

# # workaround for error: Unknown loss function:loss
from Nn.nn_functions import BasicHuberLoss
custom_loss = BasicHuberLoss(delta=1.0)
# tf.keras.losses.loss = custom_loss

tflite_model_path = os.path.join(arguments.model_path, 'converted_model.tflite')
temp_model_path = os.path.join(arguments.model_path, 'temp_model.h5')

keras_model, x_shape, y_shape = nnBuilder.build_net()
keras_model.load_weights(arguments.final_model_path)

keras_model.compile('adam','mse')

tf.keras.models.save_model(keras_model, temp_model_path)

converter = tf.contrib.lite.TocoConverter.from_keras_model_file(temp_model_path)
tflite_model = converter.convert()
open(tflite_model_path, "wb").write(tflite_model)
