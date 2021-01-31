% Calculate the gradient of the logistic regression
% objective function with respect to each parameter

function grad = LR_CalcGrad(XTrain,yTrain,wHat)

    pred = sigmoid(XTrain*wHat); %nx1
    grad = XTrain.'*(yTrain-pred);

end