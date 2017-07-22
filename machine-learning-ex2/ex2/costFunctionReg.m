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

hThetaX = sigmoid(sum((theta' .* X)'));

for i = 1 : m
  J = J  - ( y(i)*log(hThetaX(i)) + (1 - y(i))*log(1 - hThetaX(i)));
end

J = (J / m);

for i = 1 : size(J)
  if i > 1
    J(i) = J(i) + (lambda/(2*m)) * sum(theta.^2);
  end
end

for t = 1 : size(theta)

  for i = 1 : m
    grad(t) = grad(t) + (hThetaX(i) - y(i)) * X(i, t);
  end
  
  grad(t) = (grad(t) / m);
  
  if t > 1
    grad(t) = grad(t) + (lambda/m) * theta(t);
  end
end


% =============================================================

end
