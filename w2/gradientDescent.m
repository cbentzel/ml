function [theta, J_history] = gradientDescent(X, y, theta_arg, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
step = alpha / m;
theta = theta_arg

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    diffs = X*theta - y;
    grad_1 = sum(diffs.*X(:, 1));
    grad_2 = sum(diffs.*X(:, 2));
    grad = [grad_1; grad_2];
    theta = theta - step*grad;
    
    % ============================================================

    % Save the cost J in every iteration    
    cur_cost = computeCost(X, y, theta);
    J_history(iter) = cur_cost;
end

end
