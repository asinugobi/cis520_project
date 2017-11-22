function [Y_hat] = predict_labels(X_test_bag, test_raw)

% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text 
load('naive_bayes.mat') 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_test_bag); 
sparse_x(sparse_x > 1) = 1;
sparse_x = sparse_x(:, idx);

% Outputs:  Y_hat nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
Y_hat = predict(nb_model, sparse_x);  

end