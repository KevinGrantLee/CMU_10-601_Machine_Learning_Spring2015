function [error] = ClassificationError(yHat, yTruth)
    [N,~] = size(yHat);
    error = sum(yHat ~= yTruth)/N;
end

