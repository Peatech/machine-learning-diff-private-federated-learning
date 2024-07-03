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
cd MNIST_original 

# Download MNIST data using wget
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# Check if files were downloaded
if [ ! -f "train-images-idx3-ubyte.gz" ] || [ ! -f "train-labels-idx1-ubyte.gz" ] || [ ! -f "t10k-images-idx3-ubyte.gz" ] || [ ! -f "t10k-labels-idx1-ubyte.gz" ]; then
  echo "One or more MNIST files failed to download."
  exit 1
fi

# Extract the files
gunzip -f train-images-idx3-ubyte.gz
gunzip -f train-labels-idx1-ubyte.gz
gunzip -f t10k-images-idx3-ubyte.gz
gunzip -f t10k-labels-idx1-ubyte.gz

# Check if files were extracted
if [ ! -f "train-images-idx3-ubyte" ] || [ ! -f "train-labels-idx1-ubyte" ] || [ ! -f "t10k-images-idx3-ubyte" ] || [ ! -f "t10k-labels-idx1-ubyte" ]; then
  echo "One or more MNIST files failed to extract."
  exit 1
fi

cd ..
python Create_clients.py 

STRING2="You can now run differentially private federated learning on the MNIST data set. Type python sample.py —-h for help"
echo $STRING2
STRING3="An example: …python sample.py —-N 100… would run differentially private federated learning on 100 clients for a privacy budget of (epsilon = 8, delta = 0.001)"
echo $STRING3
STRING4="For more information on how to use the functions please refer to their documentation"
echo $STRING4
