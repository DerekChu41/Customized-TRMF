from __future__ import print_function
import torch, h5py
import numpy as np
from scipy.io import loadmat
from torch.nn.utils import weight_norm

import torch.nn as nn
import torch.optim as optim
import numpy as np

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

class CusomizedTRMF():

    """
    Ymat:
        Ymat is a matrix of N*T, factorize Ymat into F and X,
        which are N*rank,rank*T Matrices respectively.
    rank:
        Length of latent embedding dimension.
    size:
        how many days you want to look back to make the prediction.
    leads:
        how many days you want to predict in the future.
    alt_iter:
        Number of iterations of updating matrices F, X and Tau.
    epochs_factors:
        Number of epochs to update F and X in each iteration.
    epochs_tau:
        Number of epochs to update Tau in each iteration.
    lr_factors:
        learning rate when update matrices F and X. 0.001 recommended.
    lr_tau:
        learning rate when update Tau. 0.001 recommended.
    reg_factors:
        lambda in regularization in F and X. 0.001 recommended.
    batch:
        the batch size in pseudo mini batch. If T is quite long,
        then maybe you need to set a relative large batch
        to accelerate training speed.
    model:
        We have four different models here, "Net","LSTM","TemporalConvNet","DilationConv",
        which represent Net,LSTM,TCN and DilationConv.
        And you may customized your own model here.
        Just to make sure that the Input should be two dimensions, like X[:,i-size:i].
        And output should be one dimension,like Xout = X[:,i]
        Notice that when implement TemporalConvNet or DilationConv,
        the last element of num_channels should be 1.
    """
    def __init__(
        self,
        Ymat,
        rank,
        size,
        leads,
        alt_iter,
        epochs_factors,
        epochs_tau,
        lr_factors,
        lr_tau,
        reg_factors,
        batch,
        model
    ):
        self.Ymat = Ymat
        self.rank = rank
        self.size = size
        self.leads = leads
        self.alt_iter = alt_iter
        self.epochs_factors = epochs_factors
        self.epochs_tau = epochs_tau
        self.lr_factors = lr_factors
        self.lr_tau = lr_tau
        self.reg_factors = reg_factors
        self.batch = batch

        if model == 'Net':
            self.model = Net(size)
        elif model == 'LSTM':
            self.model = LSTMNet(size)
        elif model == 'TemporalConvNet':
            self.model = TemporalConvNet(size,[1,32,32,1])
        elif model == 'DilationConv':
            self.model = DilationConv(size,[1,32,32,1])

        self.Ymat,self.mask = self.NullDetector()
        self.Ymat,self.m,self.s,self.mini = self.normalize()
        if leads>0:
            self.test = self.Ymat[:,-self.leads:]
            self.Ymat = self.Ymat[:,:-self.leads]
            self.mask = self.mask[:,:-self.leads]

        self.F,self.X = self.SVD()

        self.Ymat = torch.from_numpy(self.Ymat).float()

    def get_model(self,A, y, lamb=0):
        """
        Regularized least-squares
        """
        n_col = A.shape[1]
        return np.linalg.lstsq(
            A.T.dot(A) + lamb * np.identity(n_col), A.T.dot(y), rcond=None
        )

    def SVD(self):

        """
        Ymat is a matrix of N*T, factorize Ymat into F and X, which are N*rank,rank*T Matrices respectively.
        """

        t0 = self.Ymat.shape[1]
        indices = np.random.choice(self.Ymat.shape[0], self.rank, replace=False)
        X = self.Ymat[indices, 0:t0]
        mX = np.std(X, axis=1)
        mX[mX == 0] = 1.0
        X = X / mX[:, None]
        Ft = self.get_model(X.transpose(), self.Ymat[:, 0:t0].transpose(), lamb=0.1)
        F = Ft[0].transpose()
        X = torch.from_numpy(X).float()
        F = torch.from_numpy(F).float()
        return F,X

    def normalize(self):

        Y = self.Ymat
        m = np.mean(Y, axis=1)
        s = np.std(Y, axis=1)
        # s[s == 0] = 1.0
        s += 1.0
        Y = (Y - m[:,None]) / s[:,None]
        mini = np.abs(np.min(Y))
        Y = Y + mini
        return Y,m,s,mini

    def step_factX_loss(self):
        """
        Set F and Tau fixed, update X with loss(Ymat, F*X)
        """

        self.X = Variable(self.X, requires_grad=True)
        optim_X = optim.Adam(params=[self.X], lr=self.lr_factors)
        Hout = torch.matmul(self.F, self.X)
        optim_X.zero_grad()
        loss = torch.mean(torch.pow(Hout - self.Ymat.detach(), 2))
        l2 = torch.mean(torch.pow(self.X, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * self.reg_factors * l2
        loss.backward()
        optim_X.step()

        return loss

    def step_factF_loss(self):
        """
        Set X and Tau fixed, update F with loss(Ymat,F*X)
        """

        self.F = Variable(self.F,requires_grad=True)
        optim_F = optim.Adam(params=[self.F], lr=self.lr_factors)

        Hout = torch.matmul(self.F, self.X)
        optim_F.zero_grad()

        loss = torch.mean(torch.pow(Hout - self.Ymat.detach(), 2))
        l2 = torch.mean(torch.pow(self.F, 2))
        r = loss.detach() / l2.detach()
        loss = loss + r * self.reg_factors * l2
        loss.backward(retain_graph=True)
        optim_F.step()

        return loss

    def step_temporal_loss_X(self):
        """
        Set F and Tau fixed, update X with temporal loss,
        which is loss(Tau(Xin),XOut), so that X could be base series.
        And We update X only here, not Tau(X).
        Notice here we feed in inputs every batch step, just works like mini-batch
        """
        for p in self.model.parameters():
            p.requires_grad = False
        T = self.X.shape[1]
        for i in range(self.size,T-1,self.batch):
            Xin = self.X[:,i-self.size:i]
            Xout = self.X[:,i]

            Xin = Variable(Xin, requires_grad=True)
            Xout = Variable(Xout, requires_grad=True)
            optim_out = optim.Adam(params=[Xout], lr=self.lr_factors)
            hatX = self.model(Xin)
            optim_out.zero_grad()
            loss = torch.mean(torch.pow(Xout - hatX.detach(), 2))
            loss.backward()
            optim_out.step()
            self.X[:,i] = Xout
        return loss

    def step_tau_loss(self):
        """
        Set F and X fixed, update Tau(X) with temporal loss,
        which is loss(Tau(Xin),XOut), so that X could be base series.
        And We update Tau(X） only here, not X.
        Notice here we feed in inputs every batch step, just works like mini-batch
        """
        optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr_tau)
        for p in self.model.parameters():
            p.requires_grad = True
        T = self.X.shape[1]
        for i in range(self.size,T-1,self.batch):
            Xin = self.X[:,i-self.size:i]
            Xout = self.X[:,i]

            Xin = Variable(Xin)
            Xout = Variable(Xout)
            optimizer.zero_grad()
            Xhat = self.model(Xin)
            criterion = nn.L1Loss()
            loss = criterion(Xout,Xhat)/torch.abs(Xhat.data).mean()
            loss.backward()
            optimizer.step()
        return loss

    def train(self):
        l_F = [0]
        l_X = [0]
        l_X_temporal = [0]
        l_tau = [0]
        for i in range(self.alt_iter):
            if i%2 == 0:
                print("training factors")
                for _ in range(self.epochs_factors):

                    l1 = self.step_factX_loss()
                    l_X = l_X + [l1.cpu().item()]
                    l1 = self.step_factF_loss()
                    l_F = l_F + [l1.cpu().item()]
                    l2 = self.step_temporal_loss_X()
                    l_X_temporal = l_X_temporal + [l2.cpu().item()]
            else:
                print("training tau model")
                for _ in range(self.epochs_tau):
                    l3 = self.step_tau_loss()
                    l_tau = l_tau + [l3.cpu().item()]
        return self.F,self.X,self.model,l_F,l_X,l_X_temporal,l_tau

    def predict_futures(self):
        """
        leads is how many days you want to predict for the next leads days.
        """
        X = self.X
        for _ in range(self.leads):
            Xin = X[:,-self.size:]
            Xout = self.model(Xin)
            X = torch.cat([X,Xout],dim=1)
        Hout = torch.matmul(self.F,X)
        return Hout[:,-self.leads:]

    def impute_missings(self):
        """Impute each missing element in timeseries.

        Model uses matrix X and F to get all missing elements.

        Parameters
        ----------

        Returns
        -------
        data : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        data = self.Ymat
        data[self.mask == 0] = torch.matmul(self.F,self.X)[self.mask == 0]
        data = data.detach().numpy()
        data = (data - self.mini)*self.s[:,None] + self.m[:,None]
        return data


    def evaluation(self,pred,test):
        mape_matrix = np.abs((pred - test)/test)
        median = np.median(mape_matrix.mean(axis=1))
        return median,mape_matrix

    def transfer2original(self,x):
        return (x-self.mini)*self.s[:,None]+self.m[:,None]

    def NullDetector(self):

        mask = np.array((~np.isnan(self.Ymat)).astype(int))

        self.Ymat[mask == 0] = 0.
        return self.Ymat,mask
