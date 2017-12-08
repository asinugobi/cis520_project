clear 
clc 

% Load data 
load('../main/train.mat')
[n, m] = size(X_train_bag); 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 

% Convert sparse matrix to binary features
sparse_x(sparse_x > 1) = 1;

tic 
model = fitcecoc(sparse_x, Y_train); 
toc 