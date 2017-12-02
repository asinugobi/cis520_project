function [cv_error] = cross_validate(test_data, train_data, kernel_type)
	[m, d] = size(train_data); 
	[p, k] = size(test_data); 

	model = svmtrain(train_data(:, d), train_data(:, 1:(d-1)), kernel_type); 
	[predicted_labels, accuracy, prob_estimates] = svmpredict(test_data(:, k), test_data(:, 1:(k-1)), model); 

	cv_error = 100 - accuracy(1); 
end