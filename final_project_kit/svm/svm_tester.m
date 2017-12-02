% Clear variables 
clear
clc 

% Load data 
load('train.mat') 

% Test SVM 
kernel = 'poly'; 
model = svm(X_train_bag, kernel); 

Y_hat = predict_labels(X_train_bag, train_raw); 
cost = performance_measure(Y_hat, Y_train); 