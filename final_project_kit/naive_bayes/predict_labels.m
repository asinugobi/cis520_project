function [Y_hat] = predict_labels(X_test_bag, test_raw)

% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text 
load('naive_bayes.mat') 

%% Preprocess the data. 
% Convert X_train_bag into feature matrix (sparse x)
sparse_x = full(X_test_bag); 
sparse_x(sparse_x > 1) = 1;

% Outputs:  Y_hat nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
[Y_hat, posterior, costs] = predict(nb_model, sparse_x);  

costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 2; 2 1 2 0 2; 2 2 2 1 0];
predictVals = [];
Y_hat = [];

% Cost sensitive multi-class loss 
for k = 1:length(posterior)
    sample = posterior(k, :)';
    for c = 1:size(costs,2)
        predictVals(c)= sample(1)*costs(1, c) + sample(2)*costs(2,c) + sample(3)*costs(3,c) + sample(4)*costs(4,c) + sample(5)* costs(5,c);
    end
   Y_hat(k,1) = find(predictVals == min(predictVals));
            
end

end