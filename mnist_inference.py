from scipy.stats import truncnorm
import numpy as np
import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def init_weights(size):
    return np.float32(truncnorm.rvs(-2, 2, size=size) * 1.0 / np.sqrt(float(size[0])))

def placeholder_inputs(batch_size):
    images_placeholder = tf.keras.Input(shape=(IMAGE_PIXELS,), batch_size=batch_size, name='images_placeholder')
    labels_placeholder = tf.keras.Input(shape=(), batch_size=batch_size, name='labels_placeholder', dtype=tf.int32)
    return images_placeholder, labels_placeholder

def mnist_fully_connected_model(images, hidden1_units, hidden2_units):
    hidden1 = tf.keras.layers.Dense(hidden1_units, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1. / np.sqrt(float(IMAGE_PIXELS))), name='hidden1')(images)
    hidden2 = tf.keras.layers.Dense(hidden2_units, activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1. / np.sqrt(float(hidden1_units))), name='hidden2')(hidden1)
    logits = tf.keras.layers.Dense(NUM_CLASSES, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=1. / np.sqrt(float(hidden2_units))), name='out')(hidden2)
    return logits

class LossLayer(tf.keras.layers.Layer):
    def call(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

class EvaluationLayer(tf.keras.layers.Layer):
    def call(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        correct = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int64), labels)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

def evaluation(logits, labels):
    return EvaluationLayer()(logits, labels)

def loss(logits, labels):
    return LossLayer()(logits, labels)

def training(loss, learning_rate):
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op
