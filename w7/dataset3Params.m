function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


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

sampleC = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sampleSigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

function error = trainAndFindError(X, y, Xval, yval, C, sigma)
         model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
         predictions = svmPredict(model, Xval);
         error = mean(double(predictions ~= yval));
end

minError = trainAndFindError(X, y, Xval, yval, sampleC(1), sampleSigma(1));
C = sampleC(1);
sigma = sampleSigma(1);

for i=1:length(sampleC),
    testC = sampleC(i);
    for j = 1:length(sampleSigma),
        testSigma = sampleSigma(j);
        error = trainAndFindError(X, y, Xval, yval, testC, testSigma);
        fprintf("Error=%f\tC = %f\sigma=%f\n", error, testC, testSigma);
        if (error < minError),
           fprintf("New min error\n");
           minError = error;
           C = testC;
           sigma = testSigma;
        end
    end
end




% =========================================================================

end
