function [w] = blrNewtonRaphsonLearn(initial_w, X, t, n_iter)
%blrNewtonRaphsonLearn learns the weight vector of 2-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_w: vector of size (D+1) x 1 where D is the number of features in
%            feature vector
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% t: vector of size N x 1 where each entry is either 0 or 1 representing
%    the true label of corresponding feature vector.
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% w: vector of size (D+1) x 1, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Adding bias to X
X = [ones(size(X,1),1) X];

for i=1:1:n_iter
    
    %Activating values of y using sigmoid function    
    y = X*initial_w;
    y = sigmoid(y);
	
    %Calculating gradient of error
	error_grad = X'*(y - t);
    
	%Calculating Hessain Matrix using sparse diagonal matrix
	RColumnVector = y.*(1-y);
    R = spdiags( RColumnVector, 0, numel(RColumnVector), numel(RColumnVector) );
    H = X'*R*X;
    HInverse = pinv(H);
    
	%Finding new weight and updating old weight
	w_new = initial_w - (HInverse*error_grad);
    initial_w = w_new;

end

w = w_new;

end
