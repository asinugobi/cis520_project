function cost= test_logistic_model(X_test_bag, test_raw, Y_test)

%%% Description: This function will test the Logistic Regression model and
%%% return an associated cost.

%% Generate predictions
Y_fit= predict_labels(X_test_bag, test_raw);

cost= performance_measure(Y_fit, Y_test);

end
