import pickle
import numpy as np
import os

def create_clients(num, dir):

    num_examples = 50000
    num_classes = 10

    # Adjusted to point to the correct directory where .npy files are stored
    data_dir = os.path.join(dir, '..', 'MNIST_original')  # Assuming MNIST_original is one level up from 'clients'

    # Check if clients file already exists
    if os.path.exists(os.path.join(dir, f'{num}_clients.pkl')):
        print(f'Client exists at: {os.path.join(dir, f"{num}_clients.pkl")}')
        return

    # Ensure directory exists
    if not os.path.exists(data_dir):  # Ensure the directory check is correct
        os.makedirs(data_dir)

    # Load data from .npy files correctly
    x_train = np.load(os.path.join(data_dir, 'train_images.npy'))  # Corrected file name and path
    y_train = np.load(os.path.join(data_dir, 'train_labels.npy'))  # Corrected file name and path

    buckets = []
    for k in range(num_classes):
        temp = []
        for j in range(int(num / 100)):
            temp = np.hstack((temp, k * num_examples/10 + np.random.permutation(int(num_examples/10))))
        buckets = np.hstack((buckets, temp))

    shards = 2 * num
    perm = np.random.permutation(shards)
    z = []
    ind_list = np.split(buckets, shards)
    for j in range(0, shards, 2):
        z.append(np.hstack((ind_list[int(perm[j])], ind_list[int(perm[j + 1])])))
        perm_2 = np.random.permutation(int(2 * len(buckets) / shards))
        z[-1] = z[-1][perm_2]

    # Save clients file
    with open(os.path.join(dir, f'{num}_clients.pkl'), 'wb') as filehandler:
        pickle.dump(z, filehandler)

    print(f'Client created at: {os.path.join(dir, f"{num}_clients.pkl")}')

if __name__ == '__main__':
    List_of_clients = [100, 200, 500, 1000, 2000, 5000, 10000]
    for j in List_of_clients:
        create_clients(j, os.path.join(os.getcwd(), 'DATA', 'clients'))
