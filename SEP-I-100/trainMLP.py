# this script trains the MLP for the IP-ISAT and SEP-ISAT preconditioning

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset 
from torch.utils.data import TensorDataset
from sklearn.cluster import KMeans
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time

use_cuda = False # at present, the training is done on CPUs
dtype = torch.float64 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

torch.set_default_dtype(torch.float64)

N = 100 # number of neurons in the hidden layers
IP_ISAT = 0 # whether to perform IP-ISAT training
SEP_ISAT = 1 # whether to perform SEP-ISAT training

class Net(nn.Module):  # define the network, 6 MLP layers with N neurons each
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11,N)
        self.fc2 = nn.Linear(N,N)
        self.fc3 = nn.Linear(N,N)
        self.fc4 = nn.Linear(N,N)
        self.fc5 = nn.Linear(N,N)
        self.fc6 = nn.Linear(N,11)
       
    def forward(self, x):
	
        x = self.fc1(x)        
        x = F.mish(x)
        
        x = self.fc2(x)        
        x = F.mish(x)
        
        x = self.fc3(x)        
        x = F.mish(x)
        
        x = self.fc4(x)        
        x = F.mish(x)
        
        x = self.fc5(x)        
        x = F.mish(x) # Mish is used as the activation function for all hidden layers
		
        x = self.fc6(x)
		
        return x

def my_loss_H(output, target, X, nClusters, nKmeans): # custom loss function for SEP-ISAT preconditioning

    output1 = output - target # calculate the residual
	
    for ii in range(nClusters):        
	
        output2 = output1[ii*nKmeans:(ii+1)*nKmeans,:] - torch.mean(output1[ii*nKmeans:(ii+1)*nKmeans,:],0,True)
        X2 = X[ii*nKmeans:(ii+1)*nKmeans,:] - torch.mean(X[ii*nKmeans:(ii+1)*nKmeans,:],0,True)
		
        grad = torch.linalg.lstsq(X2, output2,driver = 'gelsd').solution
    
        output1[ii*nKmeans:(ii+1)*nKmeans,:] = output1[ii*nKmeans:(ii+1)*nKmeans,:] - ( torch.matmul(X2,grad) + torch.mean(output1[ii*nKmeans:(ii+1)*nKmeans,:],0,True) )
		# for each K-Means cluster, subtract a least squares fit from the residual
    

    loss = (torch.mean((output1)**2)) #overall loss
    return loss       
        
        
def my_loss(output, target, X): # standard RMS loss
   loss = (torch.mean((output - target)**2))
   return loss


def train(args, model, device, train_loader, optimizer, epoch): # training for IP-ISAT networks
    model.train()
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = my_loss(output, Y, X)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.dry_run:
                break


def test(model, device, test_loader): # testing for IP-ISAT networks
    model.eval()
    test_loss = 0
    nLoss = 0
    correct = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += my_loss(output, Y, X)  # sum up batch loss
            nLoss += 1
            

    test_loss /= nLoss

    print('\nTest set: Average loss: \n',test_loss)
    
def train2(args, model, device, train_loader, optimizer, epoch, nClusters, nKmeans): # training for SEP-ISAT networks
    model.train()
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = my_loss_H(output, Y, X, nClusters, nKmeans)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            if args.dry_run:
                break


def test2(model, device, test_loader, nClusters, nKmeans): # testing for SEP-ISAT networks
    model.eval()
    test_loss = 0
    correct = 0
    nLoss = 0
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            output = model(X)
            test_loss += my_loss_H(output, Y, X, nClusters, nKmeans)  
            nLoss += 1
            

    test_loss /= nLoss

    print('\nTest set: Average loss: \n',test_loss)
    
    
def KmeansDataset(data1,n1,K1,K2,idim,device): # generate the training and testing datasets from numpy data
    # arranging the raw data via K-Means clustering on the inputs

    X = data1[:n1,:idim] # training input (from data.csv)
    Y = data1[:n1,idim:] # training output (from data.csv)
	
    kmeans = KMeans(n_clusters=K1) 
    kmeans.fit(X) # perform K-Means clustering on the input
    cl = kmeans.labels_ 
    
    d = np.column_stack((cl,X,Y)) 
    
    d = d[d[:, 0].argsort()] # arrange the training data by clusters
    
    X = d[:,1:idim+1]
    Y = d[:,idim+1:] # extract the cluster-arranged inputs and outputs
	
    X = torch.from_numpy(X).to(torch.float64)
    Y = torch.from_numpy(Y).to(torch.float64) # convert to tensors   	
    
    X, Y = X.to(device), Y.to(device) # send to device (currently, only CPU)
    
    X2 = data1[n1:,:idim]
    Y2 = data1[n1:,idim:] # testing data
	
    kmeans = KMeans(n_clusters=K2)
    kmeans.fit(X2) # perform K-Means clustering on the input
    cl = kmeans.labels_
    
    d = np.column_stack((cl,X2,Y2))
    
    d = d[d[:, 0].argsort()] # arrange the testing data by clusters
    
    X2 = d[:,1:idim+1]
    Y2 = d[:,idim+1:] # extract the cluster-arranged inputs and outputs
	
    X2 = torch.from_numpy(X2).to(torch.float64)
    Y2 = torch.from_numpy(Y2).to(torch.float64) # convert to tensors 
    
    X2, Y2 = X2.to(device), Y2.to(device) # send to device (currently, only CPU)
    
    trainDataset = TensorDataset(X,Y)
    testDataset = TensorDataset(X2,Y2) # generate the pytorch datasets

    return trainDataset, testDataset

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Preconditioned ISAT')
    parser.add_argument('--batch-size', type=int, default=500, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6000, metavar='N', #6000
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.004, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.8, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=True,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=4, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = False
    use_mps = torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        
    Nsam = 40000 # number of samples in data.csv
    
    idim = 11 # dimension of the input/output

    nKmeans = 20 # number of samples in each K-Means cluster
    
    nClusters = int (args.batch_size / nKmeans) # number of overall K-Means clusters

    K1 = int((2*Nsam/4)/nKmeans) # number of training K-Means clusters
    K2 = int((2*Nsam/4)/nKmeans) # number of testing K-Means clusters
        
    n1 = int(2*Nsam/4)

    data1 = np.genfromtxt('data.csv',delimiter=',') # get data from data.csv
    
    trainDataset, testDataset = KmeansDataset(data1,n1,K1,K2,idim,device) # format into PyTorch datasets
    
    train_loader = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=True)
    test_loader = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=True)
	# loaders for IP-ISAT training (shuffling enabled)
    
    train_loader2 = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=False)
    test_loader2 = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=False)
	# loaders for SEP-ISAT training (shuffling disabled)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer2 = optim.Adam(model.parameters(), lr=args.lr)
	
    start_time = time.time()

    scheduler = StepLR(optimizer, step_size=1000, gamma=args.gamma)
    for epoch in range(1, IP_ISAT*args.epochs + 1):  # IP-ISAT training
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        
    shuffleEpoch = 500
        
    trainDataset, testDataset = KmeansDataset(data1,n1,K1,K2,idim,device)
    train_loader2 = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=False)
    test_loader2 = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=False)
	
    scheduler2 = StepLR(optimizer, step_size=1, gamma=0.99975)

    for epoch in range(1, SEP_ISAT*args.epochs + 1): # SEP-ISAT training
	
        if (np.mod(epoch,shuffleEpoch)==1): # rearrange training/testing data into new K-Means clusters every
            # "shuffleEpoch" training epochs		
            
            trainDataset, testDataset = KmeansDataset(data1,n1,K1,K2,idim,device)
            train_loader2 = torch.utils.data.DataLoader(trainDataset,**train_kwargs,shuffle=False)
            test_loader2 = torch.utils.data.DataLoader(testDataset, **test_kwargs, shuffle=False)
            scheduler2.step()
    
        train2(args, model, device, train_loader2, optimizer2, epoch, nClusters, nKmeans)
        if (np.mod(epoch,10)==1):
            test2(model, device, test_loader2, nClusters, nKmeans)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time} seconds")

    dd = model.fc1.weight.detach().numpy() 
    dd.tofile('fc1w.csv', sep = ',')
    
    dd = model.fc2.weight.detach().numpy()    
    dd.tofile('fc2w.csv', sep = ',')
    
    dd = model.fc3.weight.detach().numpy()    
    dd.tofile('fc3w.csv', sep = ',')
    
    dd = model.fc4.weight.detach().numpy()    
    dd.tofile('fc4w.csv', sep = ',')
    
    dd = model.fc5.weight.detach().numpy()    
    dd.tofile('fc5w.csv', sep = ',')
    
    dd = model.fc6.weight.detach().numpy()    
    dd.tofile('fc6w.csv', sep = ',')
    
    dd = model.fc1.bias.detach().numpy()    
    dd.tofile('fc1b.csv', sep = ',')
    
    dd = model.fc2.bias.detach().numpy()    
    dd.tofile('fc2b.csv', sep = ',')
    
    dd = model.fc3.bias.detach().numpy()    
    dd.tofile('fc3b.csv', sep = ',')
    
    dd = model.fc4.bias.detach().numpy()    
    dd.tofile('fc4b.csv', sep = ',')
    
    dd = model.fc5.bias.detach().numpy()    
    dd.tofile('fc5b.csv', sep = ',')

    dd = model.fc6.bias.detach().numpy()    
    dd.tofile('fc6b.csv', sep = ',') # output the network weights into ASCII format

if __name__ == '__main__':
    main()
