#!/bin/sh
STRING="Downloading the MNIST-data set and creating clients"
echo $STRING

# Check if the directory exists
if [ ! -d "DiffPrivate_FedLearning" ]; then
  echo "Directory DiffPrivate_FedLearning does not exist. Please check the directory name and path."
  exit 1
fi

cd DiffPrivate_FedLearning
mkdir -p MNIST_original

# Create a Python script to download the MNIST dataset using TensorFlow/Keras
cat <<EOF > download_mnist.py
import tensorflow as tf

# Download the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Save the dataset to the MNIST_original directory
import numpy as np
np.save('MNIST_original/train_images.npy', x_train)
np.save('MNIST_original/train_labels.npy', y_train)
np.save('MNIST_original/test_images.npy', x_test)
np.save('MNIST_original/test_labels.npy', y_test)
EOF

# Run the Python script to download and save the dataset
python download_mnist.py

# Check if files were downloaded
if [ ! -f "MNIST_original/train_images.npy" ] || [ ! -f "MNIST_original/train_labels.npy" ] || [ ! -f "MNIST_original/test_images.npy" ] || [ ! -f "MNIST_original/test_labels.npy" ]; then
  echo "One or more MNIST files failed to download."
  exit 1
fi

python Create_clients.py 

STRING2="You can now run differentially private federated learning on the MNIST data set. Type python sample.py —-h for help"
echo $STRING2
STRING3="An example: …python sample.py —-N 100… would run differentially private federated learning on 100 clients for a privacy budget of (epsilon = 8, delta = 0.001)"
echo $STRING3
STRING4="For more information on how to use the functions please refer to their documentation"
echo $STRING4
