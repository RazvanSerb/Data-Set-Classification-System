% X -> the test examples for which the classes must be predicted
% weights -> the trained weights (after optimization)
% [input|hidden|output]_layer_size -> the sizes of the three layers
  
% classes -> a vector with labels from 1 to 10 corresponding to
%            the test examples given as parameter
  
function [classes] = predict_classes(X, weights, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)
  s1 = input_layer_size;
  s2 = hidden_layer_size;
  s3 = output_layer_size;

  VectorTheta1 = weights(1 : (s2 * (s1 + 1)));
  Theta1 = reshape(VectorTheta1, s2, s1 + 1);
  VectorTheta2 = weights(((s2 * (s1 + 1)) + 1) : ((s2 * (s1 + 1)) + (s3 * (s2 + 1))));
  Theta2 = reshape(VectorTheta2, s3, s2 + 1);

  [m, n] = size(X);
  classes = zeros(m, 1);
  % Calculate a1;
  a1 = [ones(1, m); X'];
  % Calculate z2;
  z2 = Theta1 * a1;
  % Calculate a2;
  a2 = [ones(1, m); sigmoid(z2)];
  % Calculate z3;
  z3 = Theta2 * a2;
  % Calculate a3;
  a3 = sigmoid(z3);
  H = a3;
  [aux, classes] = max(H, [], 1);
  classes = classes';
endfunction
