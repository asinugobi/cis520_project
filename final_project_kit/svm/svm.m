function [model] = svm(test_data, kerneltype)
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
%     m = size(test_data , 1);
%     n = size(test_data, 2);

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
%     d = size(X_train_bag, 1);
    
    sparse_x = sparse(X_train_bag); 
    sparse_x(sparse_x > 1) = 1; 
   
    
    
    % Initialize parameters
    if t == 1 
        model = svmtrain(Y_train, sparse_x, '-s 0 -t 1 -d 1 -g 1 -c 1');
    elseif t == 2
        model = svmtrain(Y_train, sparse_x, '-s 0 -t 2 -d 1 -g 1 -c 1');
    else 
        disp('Error') 
        
end
