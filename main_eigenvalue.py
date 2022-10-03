import torchvision
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import loss_calculator, compute_loss_accuracy, build_Gaussian_dataset
from utils import FullyConnected_3sigm, compute_jacobian, compute_ntk, compute_ntk_partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--i')
args = parser.parse_args()

device = 'cpu'
loss_function = 'MSE'

Ns = [1000, 2000, 3000]
ds = [int(np.sqrt(1000 * i)) for i in range(5, 1000, 50)]

for N in Ns:
    for d in ds:

        print(N, d, flush=True)

        if N > 2 * d ** 2 + d:
            min_ev = 0

        else:
            dataset_train = build_Gaussian_dataset(N, d)
            net = FullyConnected_3sigm(d).to(device)

            ntk = compute_ntk_partial(net, dataset_train, device)
            eig_values, _ = np.linalg.eigh(ntk)
            min_ev = eig_values[0]

        with open('./min_eig_partial/min_eig_' + args.i + '.txt', 'a') as f:
            f.write(str(d) + '\t' + str(N) + '\t' + str(min_ev) + '\n')
