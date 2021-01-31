% Predict the labels for a test set using logistic regression

function [yHat,numErrors] = LR_PredictLabels(XTest,yTest,wHat)

    % adding ones col for bias term
    [n,~] = size(XTest);
    newCol = ones(n,1);
    XTest = [newCol, XTest];

    pred = sigmoid(XTest*wHat); %nx1
    yHat = pred > 0.5;

    numErrors = sum(yHat ~= yTest);

end