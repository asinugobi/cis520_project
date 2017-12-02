function [pred_labels] = SVM_train(test_data, kerneltype)
    % INPUT : 
    % test_data   - m X n matrix, where m is the number of test points and n is number of features
    % kerneltype  - one of strings 'poly', 'rbf'
    %               corresponding to polynomial, and RBF kernels
    %               respectively.
    
    % OUTPUT
    % returns a m X 1 vector predicted labels for each of the test points. The labels should be +1/-1 doubles

    % Add libsvm path 
    addpath('libsvm-3.22'); 

    % Default code below. Fill in your code on all the relevant positions
    m = size(test_data , 1);
    n = size(test_data, 2);

    % store kernel type 
    if strcmp(kerneltype, 'poly')
        t = 1; 
        parameters = [1:5]; 
    elseif strcmp(kerneltype, 'rbf')
        t = 2; 
        parameters = [1e-2 1 1e1 1e2 1e3]; 
    else 
        t = 0; 
    end

    %load train_data
    load('train.mat') 

    % Store number of training points 
    d = size(X_train_bag, 1);

    %load cross-validation data

    % Do cross-validation
    % For all c
    % For all kernel parameters
    % Calculate the average cross-validation error for the 5-folds

    % Initialize parameters
    C = [1, 1e1, 1e2, 1e3, 1e4, 1e5]; 
    errors = zeros(5, length(C));
    C_optimal = zeros(5, 1); 
    cv_errors = zeros(length(parameters), 1); 

    count = 0; 

    for p = 1:length(parameters)
        p
        kernel_configs = [0 t parameters(1) C(1)];  

        % For all C, find the cross validation errors
        for i = 1:length(C)
            i
            C_test = C(i); 
            kernel_configs(4) = C_test; 
            error = zeros(5, 1);

            % Get kernel type from predetermined kernel configs 
            if t == 2 
                kernel_type = get_kernel_configs_rbf(kernel_configs); 
            else 
                kernel_type = get_kernel_configs(kernel_configs);
            end 

            % Cross validate 
            for j = 1:5
                fold = folds{j};
                cv_error = cross_validate(fold.test, fold.train, kernel_type); 
                error(j) = cv_error; 

                count = count + 1
            end

            errors(:, i) = error;

        end

        % Find the mean cross validation error for each C and determine best C
        average_error = mean(errors); 
        index_min_error = find(average_error == min(average_error)); 

        % If more than one optimal C, choose the lowest C
        if length(index_min_error) > 1
            index_min_error = index_min_error(1);
        end

        cv_errors(p) = average_error(index_min_error); 

        % Store best C value for pth parameter 
        C_optimal(p) = C(index_min_error);
    end 

    % Train SVM on training data 
    % First, find the optimal C and paramater pairing.     
    index_best_error = find(cv_errors == min(cv_errors));

    % If more than one optimal C, choose the lowest C
    if length(index_best_error) > 1
        index_best_error = index_best_error(1);
    end

    kernel_configs = [0 t 0 0];

    % Update kernel configs
    % kernel_configs(4) = C_optimal(index_best_error);
    % kernel_configs(3) = parameters(index_best_error); 
    if t == 2 
        kernel_configs(4) = 1;
        kernel_configs(3) = 1e-2;
        kernel_type = get_kernel_configs_rbf(kernel_configs); 
    else 
        kernel_configs(4) = 1;
        kernel_configs(3) = 1;
        kernel_type = get_kernel_configs(kernel_configs);
    end 

    % Return the best predicted labels 
    model = svmtrain(train(:, d), train(:, 1:d-1), kernel_type);
    [pred_labels] = svmpredict(test_data(:, n), test_data(:, 1:n-1), model);

end
