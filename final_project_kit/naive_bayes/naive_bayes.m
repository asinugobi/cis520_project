% Clear data 
clear 
clc 

% Load data 
load('../data/train.mat')
[n, m] = size(X_train_bag); 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 

% Convert sparse matrix to binary features
sparse_x(sparse_x > 1) = 1;

%% Train the model.
% Train across Naive Bayes model. 
tic 
nb_model = fitcnb(sparse_x,Y_train, 'Distribution', 'mn'); 
toc 

% Save model 
% save('naive_bayes.mat','nb_model') 

%% Test the model on training data.
% Compute the training error (loss) 
general_loss = loss(nb_model, sparse_x, Y_train)

% Generate predictions 
Y_fit = predict_labels(X_train_bag, train_raw);

% Compute expected cost from model 
cost = performance_measure(Y_fit, Y_train)

