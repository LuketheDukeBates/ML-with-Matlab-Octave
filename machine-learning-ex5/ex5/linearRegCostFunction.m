function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


lt = length(theta);
h = X * theta;

J1 = 1 / (2 * m) * (h - y)' * (h - y); 
J2 = lambda/(2 * m) * (theta(2:lt))' * theta(2:lt);
J = J1 + J2

zero_theta = theta;
zero_theta(1) = 0;

grad1 = ((1 / m) * (h - y)' * X);  
grad2 = lambda / m * zero_theta';
grad = grad1 + grad2;


% =========================================================================

grad = grad(:);

end
