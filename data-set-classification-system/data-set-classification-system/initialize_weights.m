% L_prev -> the number of units in the previous layer
% L_next -> the number of units in the next layer

% matrix -> the matrix with random values
  
function [matrix] = initialize_weights(L_prev, L_next)
  epsilon = sqrt(6) / sqrt(L_prev + L_next);
  m = L_next; n = L_prev + 1;
  a = -epsilon; b = epsilon;
  matrix = a + (b-a) * rand(m, n);
endfunction
