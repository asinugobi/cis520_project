function kernel_configs = get_kernel_configs_rbf(configs)
	svm_type = strcat('-s', {' '}, {num2str(configs(1))}); 
	kernel_type = strcat('-t', {' '}, {num2str(configs(2))}); 
	degree = strcat('-g', {' '}, {num2str(configs(3))}); 
	cost = strcat('-c', {' '}, {num2str(configs(4))});
	quiet = {'-q'}; 

	kernel_configs = strcat(svm_type, {' '}, kernel_type, {' '}, degree, {' '}, cost, {' '}, quiet); 
	kernel_configs = kernel_configs{1};
end