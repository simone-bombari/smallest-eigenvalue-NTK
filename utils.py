import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import pickle
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.multiprocessing as mpt
import os
import time


def build_Gaussian_dataset(N, d):
    X = torch.randn((N, d))
    Y = torch.randn((N, 1))

    dataset = TensorDataset(X, Y)
    
    return dataset



def call_partial_mnist(N, batch_size):
    dataset_train = torchvision.datasets.MNIST('./data/',
                                               train=True,
                                               download=True,
                                               transform=torchvision.transforms.ToTensor())

    indices = torch.arange(N)
    subset_data = Subset(dataset_train, indices)
   
    
    return dataset_train
    
    
    

class FullyConnected_4relu(nn.Module):
    def __init__(self, d):
        self.d = d
        super(FullyConnected_4relu, self).__init__()
        
        self.fc1 = nn.Linear(d, d, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='sigmoid')
        self.fc2 = nn.Linear(d, d, bias=False)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='sigmoid')
        self.fc3 = nn.Linear(d, d, bias=False)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='sigmoid')
        self.fc4 = nn.Linear(d, 1, bias=False)
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_out', nonlinearity='sigmoid')
        

    def forward(self, x):
        
        activation = F.relu  #torch.sigmoid  #
        
        x = x.view(-1, self.num_flat_features(x))
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = activation(self.fc3(x))
        x = self.fc4(x)
        return x

    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    
    
class FullyConnected_4relu_mnist(nn.Module):
    def __init__(self, n, d=784):
        self.n = n
        self.d = d
        super(FullyConnected_4relu_mnist, self).__init__()
        
        self.fc1 = nn.Linear(d, n, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='sigmoid')
        self.fc2 = nn.Linear(n, n, bias=False)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='sigmoid')
        self.fc3 = nn.Linear(n, n, bias=False)
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='sigmoid')
        self.fc4 = nn.Linear(n, 1, bias=False)
        nn.init.kaiming_normal_(self.fc4.weight, mode='fan_out', nonlinearity='sigmoid')
        

    def forward(self, x):
        
        activation = F.relu  #torch.sigmoid  #
        
        x = x.view(-1, self.num_flat_features(x))
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = activation(self.fc3(x))
        x = self.fc4(x)
        return x

    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    

class FullyConnected_3sigm(nn.Module):
    def __init__(self, d):
        self.d = d
        super(FullyConnected_3sigm, self).__init__()
        
        self.fc1 = nn.Linear(d, d, bias=False)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='sigmoid')
        self.fc2 = nn.Linear(d, d, bias=False)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='sigmoid')
        self.fc3 = nn.Linear(d, 1, bias=False)
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_out', nonlinearity='sigmoid')
        

    def forward(self, x):
        
        activation = torch.sigmoid
        
        x = x.view(-1, self.num_flat_features(x))
        x = activation(self.fc1(x))
        x = activation(self.fc2(x))
        x = self.fc3(x)
        return x

    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    

def parallel_compute_ntk(model, dataset, device):
    with mpt.Manager() as manager:
        results_dict = manager.dict()
        process = mpt.Process(target=compute_ntk, args=(model, dataset, device))
        print('parent:', os.getpid())
        process.start()
        process.join()
        return dict(results_dict)


    
    
def compute_jacobian(model, dataset, device):
    gradients = []
    model = model.to(device)
    
    prev = time.time()
    print('compute_ntk started')
    
    for x, y in DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False):
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()

        yhat = model(x)
        yhat.backward()


        g = []
        for p in model.parameters():
            if p.grad is not None:
                g.append(p.grad.reshape(-1))
        gradients.append(torch.cat(g))
    
    gradients = torch.stack(gradients)
    return gradients




def compute_ntk(model, dataset, device):
    gradients = []
    model = model.to(device)
    
    prev = time.time()
    print('compute_ntk started')
    
    for x, y in DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False):
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()

        yhat = model(x)
        yhat.backward()


        g = []
        for p in model.parameters():
            if p.grad is not None:
                g.append(p.grad.reshape(-1))
        gradients.append(torch.cat(g))
    
    
    print('time: {}. \t Gradients are computed.'.format(time.time() - prev))
    prev = time.time()
    
    with torch.no_grad():
        gradients = torch.stack(gradients)
        K = torch.matmul(gradients, gradients.T)
        kernel = K.detach().cpu().numpy()
        del K, gradients
        
        print('time: {}. \t Ntk is computed.'.format(time.time() - prev))
        
        return kernel 

    

def compute_ntk_partial(model, dataset, device):
    gradients = []
    model = model.to(device)
    
    prev = time.time()
    print('compute_ntk started')
    
    for x, y in DataLoader(dataset, batch_size=1, num_workers=0, pin_memory=True, shuffle=False):
        x = x.to(device)
        y = y.to(device)

        model.zero_grad()

        yhat = model(x)
        yhat.backward()


        g = []
        for p in model.fc2.parameters():
            if p.grad is not None:
                g.append(p.grad.reshape(-1))
        gradients.append(torch.cat(g))
    
    
    print('time: {}. \t Gradients are computed.'.format(time.time() - prev))
    prev = time.time()
    
    with torch.no_grad():
        gradients = torch.stack(gradients)
        K = torch.matmul(gradients, gradients.T)
        kernel = K.detach().cpu().numpy()
        del K, gradients
        
        print('time: {}. \t Ntk is computed.'.format(time.time() - prev))
        
        return kernel 

    


def loss_calculator(outputs, labels, loss_function, num_classes, device):
    if loss_function == 'MSE':
        criterion = nn.MSELoss()
        # targets = torch.eye(num_classes)[labels]
        # targets = targets.to(device)  # One-hot encoding
        targets = labels
    elif loss_function == 'CEL':
        criterion = nn.CrossEntropyLoss()
        targets = labels

    return criterion(outputs, targets)



def compute_loss_accuracy(data_loader, loss_function, net, num_classes, device):
    score = 0
    samples = 0
    full_loss = 0

    for input_images, labels in iter(data_loader):
        input_images, labels = input_images.to(device), labels.to(device)
        outputs = net(input_images)
        minibatch_loss = loss_calculator(outputs, labels, loss_function, num_classes, device).item()
        predicted = torch.max(outputs, 1)[1]  # Max on the first axis, in 0 we have the value of the max.

        minibatch_score = (predicted == labels).sum().item()
        minibatch_size = len(labels)  # Can be different in the last iteration
        score += minibatch_score
        full_loss += minibatch_loss * minibatch_size
        samples += minibatch_size

    loss = full_loss / samples
    accuracy = score / samples

    return loss, accuracy
