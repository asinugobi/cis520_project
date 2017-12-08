%% Logistic regression using Liblinear package
clear
clc


 %Load data
 load('../data/train.mat')
 [n,m]= size(X_train_bag);
 
 %% Preprocess the data
 % Convert the matrix into binary features
X_train_bag(X_train_bag > 1)= 1;

%% Train the logistic regression model

addpath('liblinear/') 
tic
logistic_model = train(Y_train, X_train_bag, ['-s 0', 'col']);
toc
save('logistic_model.mat', 'logistic_model');


%% Generate predictions
[Y_hat]= predict_labels(X_train_bag, train_raw);

%%
cost = performance_measure(Y_hat, Y_train)
