function kernel_configs = get_kernel_configs(configs)
	svm_type = strcat('-s', {' '}, {num2str(configs(1))}); 
	kernel_type = strcat('-t', {' '}, {num2str(configs(2))}); 
	degree = strcat('-d', {' '}, {num2str(configs(3))}); 
	gamma = '-g 1 '; 
	cost = strcat('-c', {' '}, {num2str(configs(4))});
	quiet = {'-q'}; 

	kernel_configs = strcat(svm_type, {' '}, kernel_type, {' '}, degree, {' '}, gamma, {' '}, cost, {' '}, quiet); 
	kernel_configs = kernel_configs{1};
end