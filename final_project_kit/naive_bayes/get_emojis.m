function [emoji_matrix] = get_emojis(X)

% Inputs:   X     nx10000 bag of words features

% Load data. 
load('vocabulary.mat')
vocabulary = vocabulary'; 

% Make emoji array; 
emojis = [" :)", " :(", " :/", " <3", " =)", " =]", " -_-", ...
		  " ;)", " :p", " =D", " :]", " ;D", " (:", ...
		  " :D", " ^_^", " ;(", " D:", " ):", " :-)", " ^.^", ...
		  " >:(", " :-(", " 8:"]; 
% emojis = [" :)", " :(", " :/", " <3", " =)", " =]"]; % test array
emojis = emojis'; 

% Find the index column locations of the emojis. 
idx = find(ismember(vocabulary, emojis))

% Extract and build the emoji matrix. 
emoji_matrix = X(:, idx); 

end