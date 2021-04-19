from __future__ import print_function
import torch, h5py
import numpy as np
from scipy.io import loadmat
from torch.nn.utils import weight_norm

import torch.nn as nn
import torch.optim as optim
import numpy as np

%matplotlib inline
import matplotlib.pyplot as plt
from torch.autograd import Variable

import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,n_inputs):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_inputs, 1)  # 6*6 from image dimension

        self.relu2 = nn.Sigmoid()
        self.net = nn.Sequential(
            self.fc1,
        )
    def forward(self, x):
        out = self.net(x)
        return out

class LSTMNet(nn.Module):
    def __init__(self,n_inputs):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(n_inputs, 2)
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        x = x.view(x.shape[0],1,x.shape[1])
        out,state = self.lstm(x)
        out = self.fc(out)
        return out.view(out.shape[0],1)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.1,
        init=True,
    ):
        super(TemporalBlock, self).__init__()
        self.kernel_size = kernel_size
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.init = init
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        if self.init:
            nn.init.normal_(self.conv1.weight, std=1e-3)
            nn.init.normal_(self.conv2.weight, std=1e-3)

            self.conv1.weight[:, 0, :] += (
                1.0 / self.kernel_size
            )  ###new initialization scheme
            self.conv2.weight += 1.0 / self.kernel_size  ###new initialization scheme

            nn.init.normal_(self.conv1.bias, std=1e-6)
            nn.init.normal_(self.conv2.bias, std=1e-6)
        else:
            nn.init.xavier_uniform_(self.conv1.weight)
            nn.init.xavier_uniform_(self.conv2.weight)

        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.1)

    def forward(self, x):

        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.1, init=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        self.num_channels = num_channels
        self.num_inputs = num_inputs
        self.kernel_size = kernel_size
        self.dropout = dropout

        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                    init=init,
                )
            ]

        self.network = nn.Sequential(*layers,

                                    )

    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1],1)
        out = self.network(x)
        out = out.view(out.shape[0],out.shape[2])
        return out

class DilationConv(nn.Module):
    def __init__(
        self,
        n_inputs,
        num_channels,
        dilation_size = 1,
        kernel_size=2,
        stride=1,
        dropout=0.1,
    ):
        super(DilationConv, self).__init__()
        self.kernel_size = kernel_size

        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            n_inputs = n_inputs if i == 0 else num_channels[i - 1]
            n_outputs = num_channels[i]
            layers += [nn.Conv1d(
                                n_inputs,
                                n_outputs,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=(kernel_size - 1) * dilation_size,
                                dilation=dilation_size,
                            )]

        self.network = nn.Sequential(*layers,)
        self.linear = nn.Linear(2**len(num_channels), 1)

    def forward(self, x):
        x = x.view(x.shape[0],x.shape[1],1)
        out = self.network(x)
        out = out.view(out.shape[0],out.shape[2])
        out = self.linear(out)

        return out

def get_model(A, y, lamb=0):
    """
    Regularized least-squares
    """
    n_col = A.shape[1]
    return np.linalg.lstsq(
        A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y), rcond=None
    )

def SVD(Ymat,rank):

    """
    Ymat is a matrix of N*T, factorize Ymat into F and X, which are N*rank,rank*T Matrices respectively.
    """

    t0 = Ymat.shape[1]
    indices = np.random.choice(Ymat.shape[0], rank, replace=False)
    X = Ymat[indices, 0:t0]
    mX = np.std(X, axis=1)
    mX[mX == 0] = 1.0
    X = X / mX[:, None]
    Ft = get_model(X.transpose(), Ymat[:, 0:t0].transpose(), lamb=0.1)
    F = Ft[0].transpose()
    X = torch.from_numpy(X).float()
    F = torch.from_numpy(F).float()
    return F,X

def normalize(Ymat):

    Y = Ymat
    m = np.mean(Y, axis=1)
    s = np.std(Y, axis=1)
    # s[s == 0] = 1.0
    s += 1.0
    Y = (Y - m[:,None]) / s[:,None]
    mini = np.abs(np.min(Y))
    Ymat = Y + mini
    return Ymat,m,s,mini

def step_factX_loss(Ymat,F, X, reg=0.00,lr=0.001):
    """
    Set F and Tau fixed, update X with loss(Ymat, F*X)
    """

    X = Variable(X, requires_grad=True)
    optim_X = optim.Adam(params=[X], lr=lr)
    Hout = torch.matmul(F, X)
    optim_X.zero_grad()
    loss = torch.mean(torch.pow(Hout - Ymat.detach(), 2))
    l2 = torch.mean(torch.pow(X, 2))
    r = loss.detach() / l2.detach()
    loss = loss + r * reg * l2
    loss.backward()
    optim_X.step()

    return loss

def step_factF_loss(Ymat,F, X, reg=0.00,lr=0.001):
    """
    Set X and Tau fixed, update F with loss(Ymat,F*X)
    """

    F = Variable(F,requires_grad=True)
    optim_F = optim.Adam(params=[F], lr=lr)

    Hout = torch.matmul(F, X)
    optim_F.zero_grad()

    loss = torch.mean(torch.pow(Hout - Ymat.detach(), 2))
    l2 = torch.mean(torch.pow(F, 2))
    r = loss.detach() / l2.detach()
    loss = loss + r * reg * l2
    loss.backward(retain_graph=True)
    optim_F.step()

    return loss

def step_temporal_loss_X(X, model, batch,reg=0.00,lr=0.001):
   """
   Set F and Tau fixed, update X with temporal loss,
   which is loss(Tau(Xin),XOut), so that X could be base series.
   And We update X only here, not Tau(X).


   Notice here we feed in inputs every batch step, just works like mini-batch

   """


    for p in model.parameters():
        p.requires_grad = False
    T = X.shape[1]
    for i in range(size,T-1,batch):
        Xin = X[:,i-size:i]
        Xout = X[:,i]

        Xin = Variable(Xin, requires_grad=True)
        Xout = Variable(Xout, requires_grad=True)
        optim_out = optim.Adam(params=[Xout], lr=lr)
        hatX = model(Xin)
        optim_out.zero_grad()
        loss = torch.mean(torch.pow(Xout - hatX.detach(), 2))
        loss.backward()
        optim_out.step()
        X[:,i] = Xout
    return loss

def step_tau_loss(X,model,batch,lr=0.01):
    """
    Set F and X fixed, update Tau(X) with temporal loss,
    which is loss(Tau(Xin),XOut), so that X could be base series.
    And We update Tau(X） only here, not X.


    Notice here we feed in inputs every batch step, just works like mini-batch

    """


    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    for p in model.parameters():
        p.requires_grad = True
    T = X.shape[1]
    for i in range(size,T-1,batch):
        Xin = X[:,i-size:i]
        Xout = X[:,i]

        Xin = Variable(Xin)
        Xout = Variable(Xout)
        optimizer.zero_grad()
        Xhat = model(Xin)
        criterion = nn.L1Loss()
        loss = criterion(Xout,Xhat)/torch.abs(Xhat.data).mean()
        loss.backward()
        optimizer.step()
    return loss

def train(Ymat,F,X,model,alt_iter,epochs_factors=1,epochs_tau=1,batch=1):



    l_F = [0]
    l_X = [0]
    l_X_temporal = [0]
    l_tau = [0]
    for i in range(alt_iter):

        if i%2 == 0:

#             print("training factors")
            for _ in range(epochs_factors):
                l1 = step_factF_loss(Ymat,F,X)
                l_F = l_F + [l1.cpu().item()]

                l1 = step_factX_loss(Ymat,F,X)
                l_X = l_X + [l1.cpu().item()]

                l2 = step_temporal_loss_X(X,model,batch)
                l_X_temporal = l_X_temporal + [l2.cpu().item()]
        else:

#             print("training tau model")
            for _ in range(epochs_tau):
                l3 = step_tau_loss(X,model,batch)
                l_tau = l_tau + [l3.cpu().item()]
    return F,X,model,l_F,l_X,l_X_temporal,l_tau



def predict_futures(leads,F,X,model):
    """
    leads is how many days you want to predict for the next leads days.
    """

    for _ in range(leads):
        Xin = X[:,-size:]
        Xout = model(Xin)
        X = torch.cat([X,Xout],dim=1)
    Hout = torch.matmul(F,X)
    return Hout[:,-leads:]

def impute_missings(Y,mask):
    """Impute each missing element in timeseries.

    Model uses matrix X and F to get all missing elements.

    Parameters
    ----------

    Returns
    -------
    data : ndarray, shape (n_timeseries, T)
        Predictions.
    """
    data = Y
    data[mask == 0] = torch.matmul(F,X)[mask == 0]
    return data




def evaluation(pred,truth):
    mape_matrix = np.abs((pred - test)/test)
    median = np.median(mape_matrix.mean(axis=1))
    return median,mape_matrix

def transfer2original(x,m,s,mini):
    return (x-mini)*s[:,None]+m[:,None]

def NullDetector(Y):

    mask = np.array((~np.isnan(Y)).astype(int))

    Y[mask == 0] = 0.
    return Y,mask


if __name__ == '__main__':
    """
    Ymat:
        Ymat is a matrix of N*T, factorize Ymat into F and X, which are N*rank,rank*T Matrices respectively.
    rank:
        Length of latent embedding dimension
    size:
        how many days you want to look back to make the prediction
    leads:
        how many days you want to predict in the future
    alt_iter:
        Number of iterations of updating matrices F, X and Tau.
    epochs_factors:
        Number of epochs to update F and X in each iteration
    epochs_tau:
        Number of epochs to update Tau in each iteration
    batch:
        the batch size in pseudo mini batch. If T is quite long, then maybe you need to set a relative large batch
        to accelerate training speed.
    model:
        we have four different models here, which are Net,LSTM,TCN and DilationConv.
        And you maybe customized your own model here.
        Just to make sure that the Input should be two dimensions, like X[:,i-size:i]. And output should be one dimension,
        like Xout = X[:,i]
    """

    Ymat = np.load("./datasets/electricity.npy")
    Ymat,mask = NullDetector(Ymat)
    Ymat,m,s,mini = normalize(Ymat)
    rank = 70
    size = 70
    leads = 14
    model = Net(size)

    test = Ymat[:,-leads:]
    Ymat = Ymat[:,:-leads]

    F,X = SVD(Ymat)

    alt_iter = 2
    epochs_tau = 1
    epochs_factors = 1

    Ymat = torch.from_numpy(Ymat).float()

    F,X,model,l_F,l_X,l_X_temporal,l_tau = train(Ymat,F,X,model,alt_iter,epochs_factors,epochs_tau,batch)
    predictions = predict_futures(leads,F,X,model)

    predictions = transfer2original(predictions.detach().numpy(),m,s,mini)
    test = transfer2original(test,m,s,mini)
    median,mape_matrix = evaluation(predictions,test)
    Y_imputed = impute_missings(Ymat,mask)
