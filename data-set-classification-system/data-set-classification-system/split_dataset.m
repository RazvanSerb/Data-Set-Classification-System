% X -> the loaded dataset with all training examples
% y -> the corresponding labels
% percent -> fraction of training examples to be put in training dataset
  
% X_[train|test] -> the datasets for training and test respectively
% y_[train|test] -> the corresponding labels
  
% Example: [X, y] has 1000 training examples with labels and percent = 0.85
%          -> X_train will have 850 examples
%          -> X_test will have the other 150 examples
  
function [X_train, y_train, X_test, y_test] = split_dataset(X, y, percent)
  [m, n] = size(X);
  indexes = randperm(size(X));
  X_mixed = X(indexes, :);
  y_mixed = y(indexes);
  new_m = percent * m;;
  X_train = X_mixed(1 : new_m, 1 : n);
  X_test = X_mixed((new_m + 1) : m, 1 : n);
  y_train = y_mixed(1 : new_m);
  y_test = y_mixed((new_m + 1) : m);
endfunction
