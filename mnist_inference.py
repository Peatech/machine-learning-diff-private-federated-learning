import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    images_placeholder = tf.keras.Input(shape=(IMAGE_PIXELS,), batch_size=batch_size, name='images_placeholder')
    labels_placeholder = tf.keras.Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='labels_placeholder')
    return images_placeholder, labels_placeholder

def mnist_fully_connected_model(data_placeholder, hidden1_units, hidden2_units):
    """Builds the MNIST model up to where it may be used for inference.
    Args:
    data_placeholder: Images placeholder.
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
    Returns:
    logits: Output tensor with the computed logits.
    """
    hidden1 = tf.keras.layers.Dense(hidden1_units, activation='relu')(data_placeholder)
    hidden2 = tf.keras.layers.Dense(hidden2_units, activation='relu')(hidden1)
    logits = tf.keras.layers.Dense(NUM_CLASSES)(hidden2)
    return logits

def loss(logits, labels):
    """Calculates the loss from the logits and the labels."""
    labels = tf.cast(labels, tf.int64)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def training(loss, learning_rate):
    """Sets up the training Ops."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label."""
    correct = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64))
    return tf.reduce_sum(tf.cast(correct, tf.int32))
