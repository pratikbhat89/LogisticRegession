function [error, error_grad] = mlrObjFunction(W, X, T)
% mlrObjFunction computes multi-class Logistic Regression error function 
% and its gradient.
%
% Input:
% W: the vector of size ((D + 1) * 10) x 1. Later on, it will reshape into
%    matrix of size D + 1) x 10
% X: the data matrix of size N x D
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size ((D+1) * 10) x 1 representing the gradient 
%             of error function


W = reshape(W, size(X, 2) + 1, size(T, 2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Adding bias to X
X = [ones(size(X,1),1) X];

%Calculating the activations using the logsumexp function
z = X*W;
y = logsumexp(z,2);
logY= repmat(y,1,size(W,2));
logY = z - logY;

%Calculating error (scalar value)
errorValue = T.*logY;
error = sum(errorValue(:));
error = error*(-1);

%Calculating the gradient of error
errorGrad = exp(logY) - T;
errorGrad = X'*errorGrad;
error_grad = errorGrad(:);

end
