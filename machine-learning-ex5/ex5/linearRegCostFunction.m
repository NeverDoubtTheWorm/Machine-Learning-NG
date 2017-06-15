function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
h = X*theta;
theta_reg = theta;
theta_reg(1) = 0;
z = h - y;
J_reg = theta_reg' * theta_reg * lambda;
J_cost = z' * z;
J = (J_cost + J_reg) / (2 * m);
grad_reg = lambda * theta_reg;
grad_cost = X' * z;
grad = (grad_cost + grad_reg) / m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
