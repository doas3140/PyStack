'''
    Helper functions for reading and parsing data from TFRecords
'''
import tensorflow as tf


def create_parse_fn(x_shape, y_shape):
    ''' Creates parse function for tf.data.TFRecordDataset
    @param: x shape (not including batch size) ex: [224,224,3] if img
    @param: y shape (not including batch size) ex: [224,224,3] if img
    @return parse function
    '''
    def parse_fn(serialized):
        # Define a dict with the data-names and types we expect to
        # find in the TFRecords file.
        features = {
                    'input': tf.FixedLenFeature([], tf.string),
                    'output': tf.FixedLenFeature([], tf.string),
                    # 'mask': tf.FixedLenFeature([], tf.string)
                   }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example( serialized=serialized,
                                                  features=features )
        # Get the image as raw bytes.
        x_raw = parsed_example['input']
        y_raw = parsed_example['output']
        # m_raw = parsed_example['mask']
        # Decode the raw bytes so it becomes a tensor with type.
        x = tf.decode_raw(x_raw, tf.float32)
        y = tf.decode_raw(y_raw, tf.float32)
        # m = tf.decode_raw(m_raw, tf.uint8)
        # apply transormations
        # m = tf.cast(m, tf.float32)
        # # repeat mask 2 times ex: (36,) -> (72,)
        # y = y * m
        # x = x * m
        # apply shape
        x = tf.reshape(x, x_shape)
        y = tf.reshape(y, y_shape)
        # return
        return x, y
    return parse_fn


def create_iterator( filenames, train, x_shape, y_shape, batch_size, buffer_size ):
    '''
    @param: Filenames for the TFRecords files.
    @param: Boolean whether training (True) or testing (False).
    @param: input  shape (not including batch size) ex: [224,224,3] if img
    @param: output shape (not including batch size) ex: [224,224,3] if img
    @param: Return batches of this size.
    @param: Read buffers of this size. The random shuffling
            is done on the buffer, so it must be big enough.
    '''
    # Create a TensorFlow Dataset-object which has functionality
    # for reading and shuffling data from TFRecords files.
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the x, y and m.
    dataset = dataset.map( create_parse_fn(x_shape,y_shape) )
    if train: # If training then read a buffer of the given size and randomly shuffle it.
        dataset = dataset.shuffle(buffer_size=buffer_size)
    # Get a batch of data with the given size.
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
    # Create an iterator for the dataset and the above modifications.
    iterator = dataset.make_one_shot_iterator()
    return iterator




#
