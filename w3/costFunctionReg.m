function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Is there a good way to vectorize this?
for i = 1:m,
    hyp = sigmoid(X(i, :)*theta);
    term = -y(i)*log(hyp) - (1 - y(i))*log(1 - hyp);
    J += term/m;
end
% Add regularization term.
J += (lambda/(2.0*m))*(sum(theta.^2) - theta(1)^2);

for i = 1:m,
    hyp = sigmoid(X(i, :)*theta);
    term = (hyp - y(i))/m;
    grad += term*X(i, :)';
end
% Add regularization term.
reg_grad = (lambda/m)*theta;
reg_grad(1) = 0;
grad += reg_grad;





% =============================================================

end
