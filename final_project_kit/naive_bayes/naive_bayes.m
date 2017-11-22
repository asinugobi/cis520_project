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

% Filter the features 
freq = sum(sparse_x); 
sparse_x = sparse_x(:, freq <= 750); 
new_feature_size = size(sparse_x); 

% Grab prior probabilities 
% table = tabulate(Y_train); 
% prior = (table(:,3)/100)';

%% Train the tree model.
% Train across Naive Bayes model. 
tic 
nb_model = fitcnb(sparse_x,Y_train, 'Distribution', 'mn'); 
toc 

% Compute the predicted labels from training data. 
Y_fit = predict(nb_model, sparse_x); 

% Compute the training error 
misses = Y_fit - Y_train; 
misses = length(find(misses));

error_train = misses/n * 100;

% Compute expected cost from model 
cost = performance_measure(Y_fit, Y_train);

