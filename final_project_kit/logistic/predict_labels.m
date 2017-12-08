%% Predict LR

function [Y_hat_LR]= predict_labels(X_test_bag, test_raw)

load('logistic_model.mat')

Y_test = zeros(size(X_test_bag, 1), 1); 

[predicted_label, accuracy, prob_estimates_LR]= predict(Y_test, X_test_bag, logistic_model, '-b 1 -q col');

costs = [0 3 1 2 3; 4 0 2 3 2; 1 2 0 2 1; 2 1 2 0 2; 2 2 2 1 0];
predictVals=[];
Y_hat_LR=[];

for k= 1:length(prob_estimates_LR)
    sample= (prob_estimates_LR(k, :))';
    for c= 1:size(costs,2)
        predictVals(c)= sample(1)*costs(1, c) + sample(2)*costs(5,c) + sample(3)*costs(3,c) + sample(4)*costs(4,c) + sample(5)* costs(2,c);
    end
   Y_hat_LR(k,1)= find(predictVals==min(predictVals));
            
end

end
