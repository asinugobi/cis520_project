%% k nearest neighbors 
% Load data 
load('../data/train.mat')
[n, m] = size(X_train_bag); 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_train_bag); 

% Convert sparse matrix to binary features
sparse_x(sparse_x > 1) = 1;

%% Train the KNN model.
tic 
mdl = fitcknn(sparse_x, Y_train); 
toc 

% Save the model
% save('KKN_model.mat','mdl') 

%%
% Compute the training error (loss) 
% general_loss = loss(mdl, sparse_x, Y_train); 

% Generate predictions 
Y_fit_knn = predict_labels(X_train_bag, train_raw);

% Compute expected cost from model 
cost = performance_measure(Y_fit_knn, Y_train)

