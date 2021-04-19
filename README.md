# Basic ideas
```
Y is a time series matrix, and could be factorized into two sub matrices, which are F and X.
Here we could simply treat X as the base time series,
then Y will be the linear combinations of base time series,
of course, F will provide weights.

So we define another function Tau() on X, to make sure it has the time series characteristic,
which is Tau(X_previous) = X_future.

When training, F,X and Tau will be updated alternatively.

And we could apply teacher forcing methods to predict future,
or impute missing values with this model.
```

# Parameters

```
Ymat:
    Ymat is a matrix of N*T, factorize Ymat into F and X,
    which are N*rank,rank*T Matrices respectively.
rank:
    Length of latent embedding dimension.
size:
    how many days you want to look back to make the prediction.
leads:
    how many days you want to predict in the future.
    Should be 0 if you want to impute missings.
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
```
# How to run this code?
```
A demo has been provided, details could be checked in customizedTRMF.ipynb.

Since we have a lot of hyper parameters here,
we provide a Bayesian Optimization to find the best parameter combinations.
But it is highly recommended that you feed in a default parameter combination,
so that you could have a relative good baseline.

And according to the experience, it is better to fix lr_factors,lr_tau and reg_factors
to be 0.001.
```
# Drawbacks
```
In this code, we only set Ymat[:,-leads:] as test set,
this may not be enough to evaluate the model.
A sliding window methods could be applied to make better evaluation,
but should be implemented by yourself.
```
