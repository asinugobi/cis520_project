function [Y_hat] = predict_labels(X_test_bag, test_raw)

% Inputs:   X_test_bag     nx10000 bag of words features
%           test_raw      nx1 cells containing all the raw tweets in text


% Outputs:  Y_hat           nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)

load('logistic_regression.mat')
load('Y_train.mat')


[predicted_label, accuracy, prob_estimates]= predict(Y_train, X_train_bag, logistic_model, ['-b 1', '-q', 'col']);
Y_hat= predicted_label; 

% 
% costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
% predictVals=[];
% Y_hat=[];
% 
% for k= 1:length(prob_estimates)
%     sample= (prob_estimates(k, :))';
%     for c= 1:size(costs,2)
%         predictVals(c)= sample(1)*costs(1, c) + sample(2)*costs(2,c) + sample(3)*costs(3,c) + sample(4)*costs(4,c) + sample(5)* costs(5,c);
%     end
%    Y_hat(k,1)= find(predictVals==min(predictVals));
%             
% end


end