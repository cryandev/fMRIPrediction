% Regression methods scratchpad for continuous classifiers 
% (features 1:13,29) 

% This version computes predicted values based on Kernel ridge regression
% for all features, for a constant sigma and lambda. 

% Also, determines correlation between features and ROIs to determine best
% regions for training set. 

close all
addpath('data\');
addpath('plotUtils\');
loadDisplayData(); 
solution = zeros(434,30); 

numFeatures = size(video1ratings,2); 

% Define continuous feature indices 
contClassIdx = [1:1:13,28,29]; 
numContFeatures = length(contClassIdx); 

% Pre-allocate arrays 
predCorr = zeros(1,numContFeatures); 
sigma = 4096; % Gaussian kernel width 
lambda = .05; % Ridge regression penalty factor

trainingData = [ROIsLHvideo1 ROIsRHvideo1]; 
testData = [ROIsLHvideo2(1:434,:) ROIsRHvideo2(1:434,:)]; 
trueRatings = video2ratingshalf(:,contClassIdx); 

[rows, ~] = size(testData); 
kapRows = size(trainingData,1); 
kapCols = size(testData,1); 

K = zeros(rows,rows); 
Kap = zeros(kapRows,kapCols); 
testROI = ROIsLHvideo1(:,1); 
%%
tStartOut = tic; 
for s=1:numContFeatures
    currFeature = contClassIdx(s); 
    testFeature = video1ratings(:,currFeature); 
% Find ROIs with significant correlations for each feature to build test set
    [testFeatureCorrs, maxCorr, sigIdx] = findCorrFeatures(testFeature,testROI,.5); 
% Perform Kernel ridge regression for each feature using training set
    X = trainingData; 
% X = [ROIsLHvideo1 ROIsRHvideo1];
    Y = video1ratings(:,currFeature); 
    fprintf('Computing Kernel Ridge Regression on feature %d...\n',currFeature); 
% Create matrix K: K(i,j) = K(xi,xj)
for i=1:kapRows
    for j=1:kapRows
        K(i,j) = exp(-1/(2*sigma^2)*(norm(X(i,:)-X(j,:))^2)); 
    end;
end

% Compute optimal alpha from training data
alpha = inv(K+lambda*eye(size(K)))*Y;

fprintf('Done.\n'); 

% Compute ratings for new data = testData
for i=1:kapRows
    for j=1:kapCols
        Kap(i,j) = exp(-1/(2*sigma^2)*(norm(X(i,:)-testData(j,:))^2)); 
    end
end
predictRatings = alpha'*Kap; 
predictRatings = predictRatings';
solution(:,currFeature) = predictRatings; 
predCorr(s) = corr(predictRatings,trueRatings(:,s)); 
fprintf('Feature %d Correlation: %f \n', currFeature, predCorr(s)); 
end; 
tElapsedOut = toc(tStartOut); 

fprintf('Categorized %d Features in %.2f s. \n', numContFeatures,tElapsedOut); 

% Compute correlations between predictedData and testData
for k=1:numFeatures
    contCorr(k) = corr(trueRatings(:,k),predictRatings(:,k)); 
    if isnan(contCorr(k))
        contCorr(k) = 0; 
    end; 
end; 

