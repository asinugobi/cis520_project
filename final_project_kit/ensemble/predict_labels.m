function [Y_hat, Y_hat_pre] = predict_labels(X_test_bag, test_raw)

% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text 

%% Compute ensemble method 
[Y_hat_pre, weighted_avg] = ensemble(X_test_bag);

costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
predictVals = zeros(size(costs, 2), 1); 
Y_hat = zeros(length(Y_hat_pre), 1);

% Cost sensitive multi-class loss 
for k = 1:size(weighted_avg, 1)
    sample = weighted_avg(k, :)';
 
    for c = 1:size(costs,2)
        predictVals(c)= sample(1)*costs(1, c) + sample(2)*costs(2,c) + sample(3)*costs(3,c) + sample(4)*costs(4,c) + sample(5)* costs(5,c);
    end
   Y_hat(k,1) = find(predictVals == min(predictVals));         
end

end