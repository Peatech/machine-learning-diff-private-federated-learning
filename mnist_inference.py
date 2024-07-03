import numpy as np
import math
import tensorflow as tf

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def init_weights(size):
    return np.float32(truncnorm.rvs(-2, 2, size=size) * 1.0 / math.sqrt(float(size[0])))

def inference(images, Hidden1, Hidden2):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
        biases = tf.Variable(np.zeros([Hidden1]), name='biases', dtype=tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

    with tf.name_scope('hidden2'):
        weights = tf.Variable(init_weights([Hidden1, Hidden2]), name='weights', dtype=tf.float32)
        biases = tf.Variable(np.zeros([Hidden2]), name='biases', dtype=tf.float32)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

    with tf.name_scope('out'):
        weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights', dtype=tf.float32)
        biases = tf.Variable(np.zeros([NUM_CLASSES]), name='biases', dtype=tf.float32)
        logits = tf.matmul(hidden2, weights) + biases

    return logits

def inference_no_bias(images, Hidden1, Hidden2):
    with tf.name_scope('hidden1'):
        weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
        hidden1 = tf.nn.relu(tf.matmul(images, weights))

    with tf.name_scope('hidden2'):
        weights = tf.Variable(init_weights([Hidden1, Hidden2]), name='weights', dtype=tf.float32)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights))

    with tf.name_scope('out'):
        weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights', dtype=tf.float32)
        logits = tf.matmul(hidden2, weights)

    return logits

class LossLayer(tf.keras.layers.Layer):
    def call(self, logits, labels):
        labels = tf.cast(labels, tf.int64)
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

class EvaluationLayer(tf.keras.layers.Layer):
    def call(self, logits, labels):
        correct = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64))
        return tf.reduce_sum(tf.cast(correct, tf.int32))

def placeholder_inputs(batch_size):
    images_placeholder = tf.keras.Input(shape=(IMAGE_PIXELS,), batch_size=batch_size, name='images_placeholder')
    labels_placeholder = tf.keras.Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='labels_placeholder')
    return images_placeholder, labels_placeholder

def mnist_fully_connected_model(images_placeholder, hidden1_units, hidden2_units):
    return inference_no_bias(images_placeholder, hidden1_units, hidden2_units)
