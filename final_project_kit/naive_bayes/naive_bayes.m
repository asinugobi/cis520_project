% Clear data 
clear 
clc 

% Load data 
load('../train.mat')
[n, m] = size(X_train_bag); 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 
sparse_x(sparse_x > 1) = 1;

%% Train the tree model.
% Train across Naive Bayes model. 
tic 
nb_model = fitcnb(sparse_x,Y_train, 'Distribution', 'mn'); 
toc 

% Compute the predicted labels from training data. 
% Y_fit = predict(nb_model, sparse_x); 

% Compute the training error 
general_loss = loss(nb_model, sparse_x, Y_train); 

% Compute expected cost from model 
cost = performance_measure(Y_fit, Y_train);

