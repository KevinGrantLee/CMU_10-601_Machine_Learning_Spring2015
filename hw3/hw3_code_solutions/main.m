clear; close all;
load("HW3Data.mat");

%% Train classifier on data
D = NB_XGivenY(XTrain, yTrain);
p = NB_YPrior(yTrain);

%% Predict labels 
yHatTrain = NB_Classify(D, p, XTrain);
yHatTest = NB_Classify(D, p, XTest);

%% Compute error
trainError = ClassificationError(yHatTrain, yTrain);
testError = ClassificationError(yHatTest, yTest);