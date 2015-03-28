function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha


m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % to update all thetas parallely, storing the intermediate thetas values
    % in temp variable till all values are computed
    tmp_theta=zeros(length(theta),1);
    for j=1:length(theta)

        % computing the summation of partial derivatives over theta(j)
        xx= 0;
        for i=1:m
            xx = xx + (X(i,:)*theta - y(i))*X(i,j);
        end
        tmp_theta(j) = theta(j) - xx*alpha/m;
    end
    theta = tmp_theta;

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
