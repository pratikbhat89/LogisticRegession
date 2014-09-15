function [W] = mlrNewtonRaphsonLearn(initial_W, X, T, n_iter)
%mlrNewtonRaphsonLearn learns the weight vector of multi-class Logistic
%Regresion using Newton-Raphson method
% Input:
% initial_W: matrix of size (D+1) x 10 represents the initial weight matrix 
%            for iterative method
% X: matrix of feature vector which size is N x D where N is number of
%            samples and D is number of feature in a feature vector
% T: the label matrix of size N x 10 where each row represent the one-of-K
%    encoding of the true label of corresponding feature vector
% n_inter: maximum number of iterations in Newton Raphson method
%
% Output:
% W: matrix of size (D+1) x 10, represented the learned weight obatained by
%    using Newton-Raphson method

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   YOUR CODE HERE %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Adding bias to X
X = [ones(size(X,1),1) X];

%Creating identity matrix
I = eye(size(T,2),size(T,2));

%Creating Hessain Matrix of (7160x7160)
HessianFinal = ones(size(X,2)*size(T,2),size(X,2)*size(T,2));

for i = 1:n_iter
    
    %Reshaping W to [(D+1)x10] from [((D+1)*10)x1]
    initial_W = reshape(initial_W, size(X, 2) , size(T, 2));
    
    %Calculating the activations using the logsumexp functionz = X*initial_W;
    z = X*initial_W;
	y = logsumexp(z,2);
    logy = repmat(y,1,size(initial_W,2));
    logy = z - logy;
    
    %Calculating the gradient of error
    errorGrad = exp(logy) - T;
    errorGrad = X'*errorGrad;

    %Initializing two variables for row and column index calculations
    HRowIndex = 1;
    HColumnIndex = 1;
    
       for k = 1: size(T,2)
          
           for j = 1: size(T,2)
               
               %Calculating one Hessain block of (716x716) using sparse diagonal matrix
                RColumnVector = exp(logy(:,k)).*(I(k,j)-exp(logy(:,j)));
                R = spdiags( RColumnVector, 0, numel(RColumnVector), numel(RColumnVector) );
                H = X'*R*X;
                
                %Appending one block in final hessian matrix
                HessianFinal(HRowIndex:size(X, 2)*k,HColumnIndex:size(X, 2)*j) = H;
                HColumnIndex = (size(X, 2)*j)+1;
          
           end
           
           %Updating row column indices after inner loop
           HRowIndex = (size(X, 2)*k)+1;
           HColumnIndex = 1;
          
       end
    
    %Calculating inverse of final hessain matrix   
    HInverse = pinv(HessianFinal);
    
    %Reshaping weight and error gradient
    errorGrad = errorGrad(:);
    initial_W  = initial_W(:);
    
    %Finding new weight and updating old weight
    HInverseDeltaE = (HInverse*errorGrad);
    w_new = initial_W - HInverseDeltaE;
    initial_W = w_new;
    
end

%Reshaping W to [(D+1)x10] from [((D+1)*10) x 1]
W = reshape(w_new, size(X, 2) , size(T, 2));

end

