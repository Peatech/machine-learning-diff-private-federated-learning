# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
import tensorflow as tf
import numpy as np
from scipy.stats import truncnorm
import math

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def init_weights(shape, stddev=0.1):
    """ Initialize weights with a truncated normal distribution """
    values = truncnorm(-2, 2, scale=stddev).rvs(np.prod(shape)).reshape(shape)
    return tf.Variable(values, dtype=tf.float32)

def mnist_fully_connected_model(hidden1, hidden2):
    """Builds a fully connected model with two hidden layers using tf.keras"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMAGE_PIXELS,)),  # Define input shape directly in the Input layer
        tf.keras.layers.Dense(hidden1, activation='relu', kernel_initializer=tf.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(IMAGE_PIXELS))),
        tf.keras.layers.Dense(hidden2, activation='relu', kernel_initializer=tf.initializers.TruncatedNormal(stddev=1.0 / np.sqrt(hidden1))),
        tf.keras.layers.Dense(NUM_CLASSES)
    ])
    return model

def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors."""
    images_placeholder = tf.keras.Input(shape=(IMAGE_PIXELS,), batch_size=batch_size, name='images_placeholder')
    labels_placeholder = tf.keras.Input(shape=(), batch_size=batch_size, dtype=tf.int32, name='labels_placeholder')
    return images_placeholder, labels_placeholder

class LossLayer(tf.keras.layers.Layer):
    def call(self, logits, labels):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

class EvaluationLayer(tf.keras.layers.Layer):
    def call(self, logits, labels):
        correct = tf.equal(tf.argmax(logits, axis=1), labels)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

def loss(logits, labels):
    """Calculates the loss from the logits and the labels."""
    return LossLayer()(logits, labels)

def evaluation(logits, labels):
    """Evaluate the quality of the logits at predicting the label."""
    return EvaluationLayer()(logits, labels)


