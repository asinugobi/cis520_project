% Clear data 
clear 
clc 

% Load data 
load('train.mat')
[n, m] = size(X_train_bag); 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 
sparse_x(sparse_x > 1) = 1;

%% Train the tree model.
% Train across an ensemble of 50 trees.
tree_model = TreeBagger(100, sparse_x, Y_train, 'OOBPrediction', 'On', 'Method', 'classification'); 
 
% Compute the predicted labels from training data. 
Y_fit = predict(tree_model, sparse_x); 

% Compute the training error 
Y_fit = cell2mat(Y_fit); 
Y_fit = str2num(Y_fit); 
misses = Y_fit - Y_train; 
misses = length(find(misses));

error_train = misses/n * 100;

% Compute expected cost from model 
cost = performance_measure(Y_fit, Y_train);

% Plot error across trees. 
figure;
oobErrorBaggedEnsemble = oobError(tree_model);
plot(oobErrorBaggedEnsemble)
xlabel 'Number of grown trees';
ylabel 'Out-of-bag classification error';
