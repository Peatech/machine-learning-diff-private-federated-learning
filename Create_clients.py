import pickle
import numpy as np
import os

def create_clients(num, dir):
    # Assuming the 'dir' directly refers to the path where data should be found
    # We adjust this to directly point to the MNIST_original directory
    data_dir = '/kaggle/working/machine-learning-diff-private-federated-learning/MNIST_original'

    # Check if clients file already exists
    if os.path.exists(os.path.join(dir, f'{num}_clients.pkl')):
        print(f'Client exists at: {os.path.join(dir, f"{num}_clients.pkl")}')
        return

    # Ensure MNIST_original directory exists and has the expected .npy files
    if not os.path.exists(data_dir):
        print(f"Expected MNIST data directory does not exist: {data_dir}")
        return

    # Load data from .npy files correctly
    try:
        x_train = np.load(os.path.join(data_dir, 'train_images.npy'))
        y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))
    except FileNotFoundError as e:
        print(f"Error loading .npy files: {e}")
        return

    # Rest of your client creation logic
    # ...

if __name__ == '__main__':
    List_of_clients = [100, 200, 500, 1000, 2000, 5000, 10000]
    base_dir = '/kaggle/working/machine-learning-diff-private-federated-learning/DATA/clients'
    for j in List_of_clients:
        create_clients(j, base_dir)
