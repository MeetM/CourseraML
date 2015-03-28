function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% The function gradientDescent previously implemented already covers
% the general cases of multiple fetaures.
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

end
