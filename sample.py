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
    # Updated to use the correct function from mnist_inference
    data_placeholder, labels_placeholder = mnist.placeholder_inputs(batch_size=b)
    logits = mnist.mnist_fully_connected_model(data_placeholder, hidden1, hidden2)
    loss = mnist.loss(logits, labels_placeholder)
    eval_correct = mnist.evaluation(logits, labels_placeholder)
    train_op = mnist.training(loss, learning_rate=0.001)

    # Create the global_step variable
    global_step = tf.compat.v1.Variable(0, name='global_step', trainable=False)

    # Assuming you have a function to run the differentially private federated averaging:
    Accuracy_accountant, Delta_accountant = run_differentially_private_federated_averaging(
        loss, train_op, eval_correct, DATA, data_placeholder, labels_placeholder, b=b, e=e, m=m, sigma=sigma, eps=eps,
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
    parser.add_argument('--b', type=float, default=10, help='Batches per client')
    parser.add_argument('--e', type=int, default=4, help='Epochs per client')
    parser.add_argument('--log_dir', type=str, default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed'), help='Directory to put the log data.')
    FLAGS = parser.parse_args()
    sample(FLAGS.N, FLAGS.b, FLAGS.e, FLAGS.m, FLAGS.sigma, FLAGS.eps, FLAGS.save_dir, FLAGS.log_dir)
