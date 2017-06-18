function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sig_list = [0.01 0.03 0.1 0.3 1 3 10 30];
n = length(C_list);
m = length(sig_list);
val_err = zeros(n, m);
for i=1:n
    c = C_list(i);
    for j=1:m
        sig = sig_list(j);
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, sig));
        predictions = svmPredict(model, Xval);
        val_err(i,j) = mean(double(predictions ~= yval));
    end
end
[rows, cols] = find(val_err == min(min(val_err)));
C = C_list(rows(1));
sigma = sig_list(cols(1));
% =========================================================================

end
