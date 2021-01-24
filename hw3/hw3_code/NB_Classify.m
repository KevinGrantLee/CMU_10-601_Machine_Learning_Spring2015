function [yHat] = NB_Classify(D, p, XTest)

    num = (D(1,:).*XTest + (1-D(1,:)).*(1-XTest));
    denom = (D(2,:).*XTest + (1-D(2,:)).*(1-XTest));
    log_likelihood_frac = logProd(log((num./denom).')).';
    decision_boundary = log(p/(1-p)) + log_likelihood_frac;
    yHat = (decision_boundary < 0) + 1;

end

