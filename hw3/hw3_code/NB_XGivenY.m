function [D] = NB_XGivenY(XTrain, yTrain)
    
    % MAP w/ Beta(2,1) prior
    beta_1 = 2;
    beta_2 = 1;
    
    D = [(sum(XTrain(yTrain==1, :), 1) + (beta_1 - 1)) / (sum(yTrain==1) + (beta_1+beta_2-2));
        (sum(XTrain(yTrain==2, :), 1) + (beta_1 - 1)) / (sum(yTrain==2) + (beta_1+beta_2-2))];
end

