% Run the gradient ascent algorithm for logistic regression

function [wHat,objVals] = LR_GradientAscent(XTrain,yTrain)

    % adding ones col for bias term
    [n,~] = size(XTrain);
    newCol = ones(n,1);
    XTrain = [newCol, XTrain];

    % hyperparams
    tol = 0.001;
    eta = 0.01;
   
    % initialize weights to 0
    [~,p] = size(XTrain);
    wHat = zeros(p,1);
    
    objVals = 999*ones(5000,1);
    iter = 1;
    oldObj = 0.0;
    newObj = 1.0;
    
    while ~LR_CheckConvg(oldObj,newObj,tol)
        oldObj = newObj;
        newObj = LR_CalcObj(XTrain, yTrain, wHat);
        grad = LR_CalcGrad(XTrain,yTrain,wHat);
        wHat = LR_UpdateParams(wHat,grad,eta);
        objVals(iter) = newObj;
        iter = iter+1;
    end
    
    % removing unused elements
    last_iter = find(objVals==999,1);
    objVals(last_iter:end)=[];
   
end