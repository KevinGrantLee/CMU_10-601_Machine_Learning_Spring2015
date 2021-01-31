function [ output ] = sigmoid( x )
% sigmoid function
    if x < 0
        % for numerical stability
        a = exp(x);
        output = a ./ (1 + a);
    else 
        output = 1 ./ (1 + exp(-x));
    end
end

