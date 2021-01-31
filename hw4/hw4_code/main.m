clear; close all;

% Load data
load HW4Data.mat

% Train logistic regression
[wHat,objVals] = LR_GradientAscent(XTrain,yTrain);

% Test logistic regression
[yHat,numErrors] = LR_PredictLabels(XTest,yTest,wHat);

% Print the number of misclassified examples
fprintf('Training with full train set: %d misclassified examples in the test set\n',numErrors);


%% (f)
figure; 
plot(objVals,'.-');
xlabel('Iterations'); ylabel('Objective Function'); title('Convergence Plot');
fprintf('Logistic regression converges in %d iterations.\n', length(objVals));

%% random subsets (g)
ks = 10:10:100;
Errors = zeros(length(ks),2);
for i=1:length(ks)
    % randomly sample training data
    subsetInds=randperm(length(XTrain),ks(i));
    XTrainSubset=XTrain(subsetInds,:);
    yTrainSubset=yTrain(subsetInds);
    
    [wHat,~] = LR_GradientAscent(XTrainSubset,yTrainSubset);
    
    [~,test_numErrors] = LR_PredictLabels(XTest,yTest,wHat);
    [~,train_numErrors] = LR_PredictLabels(XTrainSubset,yTrainSubset,wHat);
    Errors(i,1) = test_numErrors;
    Errors(i,2) = train_numErrors;
end

figure; hold on;
plot(ks, Errors(:,2), 'o-b');
plot(ks, Errors(:,1), 'o-r');
legend('Train Error', 'Test Error');
xlabel('Training set size'); ylabel('Error');
hold off;

%% decision boundary (i)
[~,I] = sort(wHat, 'descend');
feature_j = XTest(:, I(1));
feature_k = XTest(:, I(2));

figure; hold on;
plot(feature_j(yTest==1), feature_k(yTest==1), '.b');
plot(feature_j(yTest==0), feature_k(yTest==0), '.r');
xlabel(['X_',num2str(I(1))]); ylabel(['X_',num2str(I(2))]);

plot(feature_j, -wHat(I(1))*feature_j/wHat(I(2)), 'k');

legend('Class 1', 'Class 0', 'Decision Boundary');

hold off;
