% Clear data 
clear 
clc 

% Load data 
load('../main/train.mat')
[n, m] = size(X_train_bag); 


%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 
% sparse_x(sparse_x > 1) = 1;

freq = sum(sparse_x);
sparse_x = sparse_x(:, freq <= 100 & freq >= 10);

%% Train the logistic regression model 
[B, dev, stats] = mnrfit(sparse_x, Y_train); 

% Compute the predicted probabalities from training data. 
pihat = mnrval(B, sparse_x); 

% Compute the predicted labels from the training data. 
[val, Y_fit] = max(pihat);
Y_fit = Y_fit'; 

% Compute the training error 
misses = Y_fit - Y_train; 
misses = length(find(misses));

error_train = misses/n * 100;

% Compute expected cost from model 
cost = performance_measure(Y_fit, Y_train);