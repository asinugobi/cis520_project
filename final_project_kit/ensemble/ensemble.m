function [Y_hat, weighted_avg] = ensemble(X_test_bag)
	% Inputs:   X_test_bag     nx10000 bag of words features
	
	% Load data 
	load('naive_bayes.mat')
	load('logistic_regression.mat') 

	%% Preprocess the data. 
	% Convert X_train_bag into feature matrix (sparse x)
	sparse_x = full(X_test_bag); 
	sparse_x(sparse_x > 1) = 1;
	Y_test = zeros(size(X_test_bag, 1), 1);

	% Outputs:  Y_hat nx1 predicted labels (1 for joy, 2 for sadness, 3 for surprise, 4 for anger, 5 for fear)
	[Y_hat_nb, probs_nb, costs_model] = predict(nb_model, sparse_x);  
	[Y_hat_lr, accuracy, probs_lr] = predict(Y_test, X_test_bag, logistic_model, '-b 1 -q col');

	% Swap columns in probabilities of logistic regression
	temp = probs_lr(:, 2); 
	probs_lr(: , 2) = probs_lr(:, 5); 
	probs_lr(:, 5) = temp; 

	% Calculated weighted average for each sample 
	weighted_avg = (probs_lr + probs_nb)./2; 

	% Calculate new prediction labels
	Y_hat = zeros(size(X_test_bag, 1), 1); 

	for i = 1:length(Y_hat)
		[val, idx] = max(weighted_avg(i, :)); 
		Y_hat(i) = idx; 
	end

end

