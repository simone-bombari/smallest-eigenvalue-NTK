import torchvision
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import loss_calculator, compute_loss_accuracy, build_Gaussian_dataset
from utils import FullyConnected_4relu, compute_ntk

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--Ns')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device, flush=True)

device='cpu'

loss_function = 'MSE'

possible_Ns = [[400 * i for i in range(1, 8)],
               [400 * i for i in range(9, 15)],
               [400 * i for i in range(15, 20)],
               [400 * i for i in range(20, 24)],
               [400 * i for i in range(24, 28)],
               [400 * i for i in range(28, 31)],
              ]

Ns = possible_Ns[int(args.Ns) % 6]
ds = [int(np.sqrt(500 * i)) for i in range(1, 31)]

for N in Ns:
    for d in ds:
        
        print(N, d)
        
        dataset_train = build_Gaussian_dataset(N, d)
        net = FullyConnected_4relu(d).to(device)
        
        batch_size = N
        lr = 1 / (10 * np.sqrt(N))

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        epochs = 10000
        previous_loss = np.inf

        for epoch in range(epochs):

            if epoch % 100 == 0:
                if epoch > 0:
                    if average_loss > previous_loss:
                        break
                    else:
                        previous_loss = average_loss
                average_loss = 0
                
            total_labels = np.zeros(10)
            for input_images, labels in iter(train_loader):
                input_images, labels = input_images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(input_images)
                loss = loss_calculator(outputs, labels, loss_function, None, device)
                loss.backward()
                optimizer.step()

            if epoch % 100 == 99:
                with torch.no_grad():
                    print('Epoch {}:\nTrain loss = {:.7f}\n'.format(
                           epoch+1, loss), flush=True)
                    
            average_loss += loss
                
        previous_loss = previous_loss.item() / 50

        with open('./early_stopping_diag_cpu/optimization_' + args.Ns + '.txt', 'a') as f:
            f.write(str(d) + '\t' + str(N) + '\t' + str(previous_loss) + '\n')
