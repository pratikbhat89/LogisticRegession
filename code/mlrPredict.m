function [label] = mlrPredict(W, X)
% blrObjFunction predicts the label of data given the data and parameter W
% of multi-class Logistic Regression
%
% Input:
% W: the matrix of weight of size (D + 1) x 10
% X: the data matrix of size N x D
%
% Output: 
% label: vector of size N x 1 representing the predicted label of
%        corresponding feature vector given in data matrix

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Adding bias to X
X = [ones(size(X,1),1) X];

%Creating label vector
label = zeros(size(X,1),1);

%Output values after applying activation
z = X*W;
y = logsumexp(z,2);
logY = repmat(y,1,size(W,2));
label_oneOfK = z - logY;

[C,I] = max(label_oneOfK,[],2);

%Predicting the label for each image
for i=1:size(X,1)
    label(i,1) = I(i);
end

end

