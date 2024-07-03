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

    # Initialize placeholders, ensure batch size is an integer
    data_placeholder, labels_placeholder = mnist.placeholder_inputs(batch_size=int(b))

    # Initialize a TensorFlow model using tf.keras here:
    model = mnist.mnist_fully_connected_model(hidden1, hidden2)

    # Compile the model
    model = mnist.training(model, mnist.loss)

    # Define eval_correct and loss
    logits = model(data_placeholder)
    loss = mnist.loss(logits, labels_placeholder)
    eval_correct = mnist.evaluation(logits, labels_placeholder)

    # Run differentially private federated averaging
    Accuracy_accountant, Delta_accountant = run_differentially_private_federated_averaging(
        loss, model.train_on_batch, eval_correct, DATA, data_placeholder, labels_placeholder, 
        b=int(b), e=e, m=m, sigma=sigma, eps=eps, save_dir=save_dir, log_dir=log_dir
    )

def main(_):
    sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e, m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps, save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir)

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
