function [error, error_grad] = blrObjFunction(w, X, t)
% blrObjFunction computes 2-class Logistic Regression error function and
% its gradient.
%
% Input:
% w: the weight vector of size (D + 1) x 1 
% X: the data matrix of size N x D
% t: the label vector of size N x 1 where each entry can be either 0 or 1
%    representing the label of corresponding feature vector
%
% Output: 
% error: the scalar value of error function of 2-class logistic regression
% error_grad: the vector of size (D+1) x 1 representing the gradient of
%             error function


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Adding bias to X
X = [ones(size(X,1),1) X];

%Activating values of y using sigmoid function  
y = X*w;
y = sigmoid(y);

%Calculating error (scalar value)
errorValue = t.*log(y) + (1-t).*log(1-y);
error = sum(errorValue(:));
error = error*(-1);

%Calculating gradient of error
error_grad = X'*(y - t);

end
