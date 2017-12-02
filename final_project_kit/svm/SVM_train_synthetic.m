function [pred_labels, C_optimal, train_errors, test_errors, cv_errors, models] = SVM_train(test_data, kerneltype)
    % INPUT : 
    % test_data   - m X n matrix, where m is the number of test points and n is number of features
    % kerneltype  - one of strings 'poly', 'rbf'
    %               corresponding to polynomial, and RBF kernels
    %               respectively.
    
    % OUTPUT
    % returns a m X 1 vector predicted labels for each of the test points. The labels should be +1/-1 doubles

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
    datadir = 'Synthetic/';
    load(strcat(datadir,'train.mat'));

    %load cross-validation data
    folds = cell(5,1); 

    fold1.train = load('Synthetic/CrossValidation/Fold1/cv-train.mat'); 
    fold1.train = fold1.train.cv_train; 
    fold1.test = load('Synthetic/CrossValidation/Fold1/cv-test.mat'); 
    fold1.test = fold1.test.cv_test; 

    fold2.train = load('Synthetic/CrossValidation/Fold2/cv-train.mat'); 
    fold2.train = fold2.train.cv_train; 
    fold2.test = load('Synthetic/CrossValidation/Fold2/cv-test.mat'); 
    fold2.test = fold2.test.cv_test; 

    fold3.train = load('Synthetic/CrossValidation/Fold3/cv-train.mat'); 
    fold3.train = fold3.train.cv_train;  
    fold3.test = load('Synthetic/CrossValidation/Fold3/cv-test.mat'); 
    fold3.test = fold3.test.cv_test;  

    fold4.train = load('Synthetic/CrossValidation/Fold4/cv-train.mat'); 
    fold4.train = fold4.train.cv_train; 
    fold4.test = load('Synthetic/CrossValidation/Fold4/cv-test.mat'); 
    fold4.test = fold4.test.cv_test;  

    fold5.train = load('Synthetic/CrossValidation/Fold5/cv-train.mat'); 
    fold5.train = fold5.train.cv_train; 
    fold5.test = load('Synthetic/CrossValidation/Fold5/cv-test.mat'); 
    fold5.test = fold5.test.cv_test; 

    folds{1} = fold1; 
    folds{2} = fold2; 
    folds{3} = fold3; 
    folds{4} = fold4; 
    folds{5} = fold5; 

    % Do cross-validation
    % For all c
    % For all kernel parameters
    % Calculate the average cross-validation error for the 5-folds

    % Initialize parameters
    C = [1, 1e1, 1e2, 1e3, 1e4, 1e5]; 
    errors = zeros(5, length(C));
    C_optimal = zeros(5, 1); 
    models = cell(5, 1); 
    train_errors = zeros(length(parameters), 1);
    test_errors = zeros(length(parameters), 1);
    all_labels = zeros(m, length(parameters)); 
    cv_errors = zeros(length(parameters), 1); 

    for p = 1:length(parameters)
        p
        kernel_configs = [0 t parameters(p) C(1)];  

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

        %Train SVM on training data 
        kernel_configs(4) = C_optimal(p); 
    
        if t == 2 
            kernel_type = get_kernel_configs_rbf(kernel_configs); 
        else 
            kernel_type = get_kernel_configs(kernel_configs);
        end 

        model = svmtrain(train(:, 3), train(:, 1:2), kernel_type); 

        % Store best model for pth parameter
        models{p} = model; 

        % Calculate and store training and testing errors
        [pred_labels_train, accuracy_train, prob_estimates] = svmpredict(train(:, 3), train(:, 1:2), model); 
        [pred_labels_test, accuracy_test, prob_estimates] = svmpredict(test_data(:, 3), test_data(:, 1:2), model);
        train_errors(p) = 100 - accuracy_train(1);
        test_errors(p) = 100 - accuracy_test(1);

        all_labels(:, p) = pred_labels_test; 
    end

    % Return the best predicted labels 
    index_best_model = find(test_errors == min(test_errors)); 
    pred_labels = all_labels(index_best_model); 
end
