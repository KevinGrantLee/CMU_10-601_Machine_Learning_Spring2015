clear; close all;
load("HW3Data.mat");

%% When using small datasets in (h)
% XTrain = XTrainSmall;
% yTrain = yTrainSmall;

%% Train classifier on data
D = NB_XGivenY(XTrain, yTrain);
p = NB_YPrior(yTrain);

%% Predict labels 
yHatTrain = NB_Classify(D, p, XTrain);
yHatTest = NB_Classify(D, p, XTest);

%% Compute error
trainError = ClassificationError(yHatTrain, yTrain);
testError = ClassificationError(yHatTest, yTest);

%% Visualize 5 most common words for each class
% class 1

[~,I1] = sort(D(1,:), 'descend');
class1commonwords = Vocabulary(I1(1:5));
[~,I1] = sort(D(1,:)./D(2,:), 'descend');
class1maxwords = Vocabulary(I1(1:5));

% class 2
[~,I2] = sort(D(2,:), 'descend');
class2commonwords = Vocabulary(I2(1:5));
[~,I2] = sort(D(2,:)./D(1,:), 'descend');
class2maxwords = Vocabulary(I2(1:5));