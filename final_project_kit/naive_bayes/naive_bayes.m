% Clear data 
clear 
clc 

% Load data 
load('train.mat')
[n, m] = size(X_train_bag); 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 

% Grab emojis 
emoji_matrix = get_emojis(sparse_x);

% Convert sparse matrix to binary features
sparse_x(sparse_x > 1) = 1;

% Add additional emoji counts to feature matrix 
sparse_x(:, m+1:m+size(emoji_matrix, 2)) = emoji_matrix; 

%% Train the tree model.
% Train across Naive Bayes model. 
tic 
nb_model = fitcnb(sparse_x,Y_train, 'Distribution', 'mn'); 
toc 

% Compute the training error (loss) 
general_loss = loss(nb_model, sparse_x, Y_train); 

% Generate predictions 
Y_fit = predict_labels(X_train_bag, train_raw);

% Compute expected cost from model 
cost = performance_measure(Y_fit, Y_train);

