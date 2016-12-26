function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
step = alpha/m;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    diffs = X*theta - y;
    for i = 1:size(X, 2),
      grad(i, :) = sum(diffs.*X(:, i));
    end;
    theta = theta - step*grad;

    % ============================================================

    % Save the cost J in every iteration
    cur_cost = computeCostMulti(X, y, theta);
    J_history(iter) = cur_cost;
end

end
