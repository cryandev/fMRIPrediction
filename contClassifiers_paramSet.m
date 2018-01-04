% Regression methods scratchpad for continuous classifiers 
% (features 1:13,29) 

% This version computes predicted values based on Kernel ridge regression,
% for a single feature and array of values for sigma and lambda, to 
% determine optimal values for each. 

% Determine correlation between features and ROIs to determine best
% training set

close all
addpath('data\');
load ratings.mat
load ROITimeseries.mat

numFeatures = size(video1ratings,2); 
contClassIdx = [1:1:13,28,29]; 
numContFeatures = length(contClassIdx); 

currFeatureIdx = 6; 
currFeature = contClassIdx(currFeatureIdx); 

% Array of values to explore 
sigma = 2.^(7:.5:9); % Gaussian kernel width 
lambda = .01:.01:.1; % Ridge regression penalty factor

sigSize = length(sigma); 
lamSize = length(lambda);
predCorr = zeros(lamSize,sigSize); 

trainingData = [ROIsLHvideo1 ROIsRHvideo1]; 
testData = [ROIsLHvideo2(1:434,:) ROIsRHvideo2(1:434,:)]; 
trueRatings = video2ratingshalf(:,contClassIdx); 
testFeature = video1ratings(:,currFeature); 

[rows, ~] = size(testData); 
kapRows = size(trainingData,1); 
kapCols = size(testData,1); 

K = zeros(rows,rows); 
Kap = zeros(kapRows,kapCols); 

% Perform Kernel ridge regression for each feature using training set
X = trainingData; 
Y = video1ratings(:,currFeature); 
[kapRows, cols] = size(X); 

tStartOut = tic; 

for l=1:lamSize
    for s=1:sigSize
fprintf('Computing (l,s) pair = (%.3f, %.3f) on feature %d...\n',... 
            lambda(l),sigma(s),currFeature); 

% Create matrix K: K(i,j) = K(xi,xj)
for i=1:kapRows
    for j=1:kapRows
        K(i,j) = exp(-1/(2*sigma(s)^2)*(norm(X(i,:)-X(j,:))^2)); 
    end;
end
% Compute optimal alpha from training data
alpha = (K+lambda(l)*eye(size(K)))\Y;
fprintf('Done.\n'); 

% Compute ratings for new data = testData
for i=1:kapRows
    for j=1:kapCols
        Kap(i,j) = exp(-1/(2*sigma(s)^2)*(norm(X(i,:)-testData(j,:))^2)); 
    end
end
predictRatings = alpha'*Kap; 
predictRatings = predictRatings';
predCorr(l,s) = corr(predictRatings,trueRatings(:,currFeatureIdx)); 
fprintf('Feature %d Correlation: %f \n', currFeature, predCorr(s)); 
    end;
end; 
tElapsedOut = toc(tStartOut); 

% Plot results
fprintf('(s,l) run completed on feature %d in %.4f s. \n', currFeature,tElapsedOut); 
[maxCorr maxIdx] = max(predCorr); 
fprintf('Max correlation of %.4f at (l,s) = (%.3f, %.3f)\n', maxCorr, lambda,sigma(maxIdx)); 
%
close all
idx = 1:1:length(predictRatings); 
figure
h1 = subplot(2,2,1); plot(sigma,predCorr,'-*k'); a1 = gca; axis tight
h2 = subplot(2,2,2); plot(lambda,predCorr,'-*k'); a2 = gca; 
h3 = subplot(2,2,[3 4]); plot(idx,LPpredictRatings,'k',idx,trueRatings(:,currFeatureIdx)); 
plot(idx,predictRatings,'k',idx,trueRatings(:,currFeatureIdx)); a3 = gca; 

xlabel(a1, '\lambda','FontWeight','Bold'); 
ylabel(a1, 'Correlation'); 

xlabel(a2,'\lambda','FontWeight','Bold'); 
ylabel(a2, 'Correlation'); 

xlabel(a3,'Time'); 
ylabel(a3, 'Rating'); 

title(h1,['Feature: ', num2str(currFeature)]); 
title(h2,['Feature: ', num2str(currFeature)]); 

