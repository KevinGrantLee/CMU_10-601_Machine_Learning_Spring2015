function [p] = NB_YPrior(yTrain)
    [N,~] = size(yTrain);
    p = sum(yTrain==1) / N;
end

