function cost = test_nb_model(X_test_bag, test_raw, Y_test)
    %% Description 
    % This function will test the Naive Bayes model and return an 
    % associated cost. 

    % Generate predictions 
    Y_fit = predict_labels(X_test_bag, test_raw);

    % Compute expected cost from model 
    cost = performance_measure(Y_fit, Y_test)
end 