% path -> a relative path to the .mat file that must be loaded

% X, y -> the training examples (X) and their corresponding labels (y)

function [X, y] = load_dataset(path)
  load(path);
endfunction
