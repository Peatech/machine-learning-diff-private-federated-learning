import tensorflow as tf
import mnist_inference as mnist  # Ensure mnist_inference is updated for TensorFlow 2.x
import os
from DiffPrivate_FedLearning import run_differentially_private_federated_averaging
from MNIST_reader import Data
import argparse
import sys

def sample(N, b, e, m, sigma, eps, save_dir, log_dir):
    # Specs for the model that we would like to train in differentially private federated fashion:
    hidden1 = 600
    hidden2 = 100

    # DATA is expected to be an object with client structures and training examples
    DATA = Data(save_dir, N)

    # Initialize a TensorFlow model using tf.keras here:
    data_placeholder, labels_placeholder = mnist.placeholder_inputs(batch_size=int(b))
    logits = mnist.mnist_fully_connected_model(data_placeholder, hidden1, hidden2)

    # Create the global_step variable
    global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)

    # Wrap loss computation in a custom layer
    class LossLayer(tf.keras.layers.Layer):
        def call(self, logits, labels):
            labels = tf.cast(labels, tf.int64)
            return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    loss_layer = LossLayer()
    loss = loss_layer(logits, labels_placeholder)

    # Wrap evaluation in a custom layer
    class EvaluationLayer(tf.keras.layers.Layer):
        def call(self, logits, labels):
            correct = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, tf.int64))
            return tf.reduce_sum(tf.cast(correct, tf.int32))

    eval_layer = EvaluationLayer()
    eval_correct = eval_layer(logits, labels_placeholder)

    # Set up the training operation
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Assuming you have a function to run the differentially private federated averaging:
    Accuracy_accountant, Delta_accountant = run_differentially_private_federated_averaging(
        loss, train_op, eval_correct, DATA, data_placeholder, labels_placeholder, b=int(b), e=e, m=m, sigma=sigma, eps=eps,
        save_dir=save_dir, log_dir=log_dir
    )

def main(_):
    sample(FLAGS.N, FLAGS.b, FLAGS.e, FLAGS.m, FLAGS.sigma, FLAGS.eps, FLAGS.save_dir, FLAGS.log_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default=os.getcwd(), help='directory to store progress')
    parser.add_argument('--N', type=int, default=100, help='Total Number of clients participating')
    parser.add_argument('--sigma', type=float, default=0, help='The gm variance parameter; will not affect if Priv_agent is set to True')
    parser.add_argument('--eps', type=float, default=8, help='Epsilon')
    parser.add_argument('--m', type=int, default=0, help='Number of clients participating in a round')
    parser.add_argument('--b', type=int, default 10, help='Batches per client')
    parser.add_argument('--e', type=int, default=4, help='Epochs per client')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed'), help='Directory to put the log data.')
    FLAGS = parser.parse_args()
    sample(FLAGS.N, FLAGS.b, FLAGS.e, FLAGS.m, FLAGS.sigma, FLAGS.eps, FLAGS.save_dir, FLAGS.log_dir)
