function [Y_hat] = predict_labels(X_test_bag, test_raw)
% Inputs:   X_test_bag     nx9995 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text 
load('treebagger.mat') 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
X_test_bag(:, 9996:10000) = 0;
sparse_x = full(X_test_bag); 
sparse_x(sparse_x > 1) = 1;

% Outputs:  Y_hat nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
Y_hat = predict(tree_model, sparse_x); 
Y_hat = cell2mat(Y_hat); 
Y_hat = str2num(Y_hat); 
end