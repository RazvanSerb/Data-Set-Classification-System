% params -> vector containing the weights from the two matrices
%           Theta1 and Theta2 in an unrolled form (as a column vector)
% X -> the feature matrix containing the training examples
% y -> a vector containing the labels (from 1 to 10) for each
%      training example
% lambda -> the regularization constant/parameter
% [input|hidden|output]_layer_size -> the sizes of the three layers
  
% J -> the cost function for the current parameters
% grad -> a column vector with the same length as params
% These will be used for optimization using fmincg

function [J, grad] = cost_function(params, X, y, lambda, ...
                   input_layer_size, hidden_layer_size, ...
                   output_layer_size)
  s1 = input_layer_size;
  s2 = hidden_layer_size;
  s3 = output_layer_size;
  % Get Theta1 and Theta2 (from params). Hint: reshape
  Theta1 = reshape(params(1 : (s2 * (s1 + 1))), s2, s1 + 1);
  Theta2 = reshape(params(((s2 * (s1 + 1)) + 1) : ((s2 * (s1 + 1)) + (s3 * (s2 + 1)))), s3, s2 + 1);
  % Forward propagation
  [m, n] = size(X);
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
  % Compute the error in the output layer and perform backpropagation
  Delta1 = zeros(s2, s1 + 1);
  Delta2 = zeros(s3, s2 + 1);
  Y = zeros(s3, m);
  Y(sub2ind([s3, m], y', 1 : m)) = 1;
  error3 = a3 - Y;
  Delta2 = Delta2 + error3 * (a2');
  error2 = (Theta2') * error3 .* (a2 .* (1 - a2));
  error2 = error2(2 : (s2 + 1), :);
  Delta1 = Delta1 + error2 * (a1');
  % Determine J and the gradients
  suma = - Y .* log(H) - (1 - Y) .* log(1 - H);
  J = sum(suma(:)) / m;
  J = J + (lambda / (2*m)) * (norm(Theta1(:, 2 : (s1+1)), 'fro') ^ 2);
  J = J + (lambda / (2*m)) * (norm(Theta2(:, 2 : (s2+1)), 'fro') ^ 2);
  VectorDelta1 = reshape(Delta1, (s2 * (s1 + 1)), 1);
  VectorTheta1 = [zeros(s2, 1); reshape(Theta1(:, 2 : (s1 + 1)), [], 1)];
  VectorGradient1 = (VectorDelta1 + lambda * VectorTheta1) / m;
  VectorDelta2 = reshape(Delta2, (s3 * (s2 + 1)), 1);
  VectorTheta2 = [zeros(s3, 1); reshape(Theta2(:, 2 : (s2 + 1)), [], 1)];
  VectorGradient2 = (VectorDelta2 + lambda * VectorTheta2) / m;
  grad = [VectorGradient1; VectorGradient2];
endfunction
