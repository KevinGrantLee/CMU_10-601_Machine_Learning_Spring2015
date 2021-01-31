% Calculate the logistic regression objective value

function obj = LR_CalcObj(XTrain,yTrain,wHat)

    pred = sigmoid(XTrain*wHat); %nx1
    % need to check this
    obj = sum( yTrain.*log(pred) + (1-yTrain).*log(1-pred) );
  
end