%% Treebagger tester 

%% Clear variables 
clc 
clear 

%% Load data 
load('train.mat')

tic 
Y_fit = predict_labels(X_train_bag(:, 1:9995), train_raw); 
cost = performance_measure(Y_fit, Y_train); 
toc