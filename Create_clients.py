import pickle
import numpy as np
import os

def create_clients(num, dir):
    # Assuming the 'dir' is the base directory where the client data should be stored
    data_dir = os.path.join(dir, 'DATA', 'clients')

    # Ensure the 'clients' directory exists
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")

    # Path for client data file
    client_file_path = os.path.join(data_dir, f'{num}_clients.pkl')

    # Check if clients file already exists
    if os.path.exists(client_file_path):
        print(f'Client data already exists at: {client_file_path}')
        return

    # Load the MNIST data from the MNIST_original directory
    mnist_data_dir = os.path.join(dir, 'MNIST_original')
    if not os.path.exists(mnist_data_dir):
        print(f"Expected MNIST data directory does not exist: {mnist_data_dir}")
        return

    try:
        x_train = np.load(os.path.join(mnist_data_dir, 'train_images.npy'))
        y_train = np.load(os.path.join(mnist_data_dir, 'train_labels.npy'))
    except FileNotFoundError as e:
        print(f"Error loading .npy files: {e}")
        return

    num_examples = x_train.shape[0]
    num_classes = len(np.unique(y_train))

    buckets = []
    for k in range(num_classes):
        class_indices = np.where(y_train == k)[0]
        np.random.shuffle(class_indices)
        temp = np.array_split(class_indices, num // num_classes)
        buckets.extend(temp)

    # Randomly distribute remaining data points to different clients (for non-iid)
    extra_indices = np.setdiff1d(np.arange(num_examples), np.concatenate(buckets))
    np.random.shuffle(extra_indices)
    extra_splits = np.array_split(extra_indices, num)

    clients = [np.concatenate((buckets[i], extra_splits[i])) for i in range(num)]
    clients = [client.astype(int) for client in clients]  # ensure indices are integers

    # Save clients file
    with open(client_file_path, 'wb') as filehandler:
        pickle.dump(clients, filehandler)
    print(f'Client data saved at: {client_file_path}')

if __name__ == '__main__':
    base_dir = '/kaggle/working/machine-learning-diff-private-federated-learning'
    List_of_clients = [100, 200, 500, 1000, 2000, 5000, 10000]
    for j in List_of_clients:
        create_clients(j, base_dir)
