% fMRI prediction main training model 

% Top-level script processing that loads data and calls training methods 
% for prediction of the second half of video 2 ratings. 
% Training data set is taken from the entirety of the ratings and brain 
% ROI data from video 1. 

clear
% Add data path to Matlab search path; load relevant files 
addpath('data')

% toggle save results 
saveRes = 0; 

% Load data
load ratings.mat
load ROITimeseries.mat

numFeatures = size(video1ratings,2); 
numPredTimeSteps = size(video2ratingshalf,1); 
predRatings = zeros(numPredTimeSteps,numFeatures); 

% Define training set  
trainingData = [ROIsLHvideo1 ROIsRHvideo1]; 
testData = [ROIsLHvideo2(1:numPredTimeSteps,:), ...
            ROIsRHvideo2(1:numPredTimeSteps,:)]; 

testFeature = video1ratings; 
trueRatings = video2ratingshalf; 

%------------------------------------------------------------------------%
% Binary classifier indices 
binClassIdx = [14:1:27,30]; 
numBinFeatures = length(binClassIdx); 

% Continuous classifier indices
contClassIdx = [1:1:13,28,29]; 
numContFeatures = length(contClassIdx); 

% Define best method to apply to each binary feature - 
% 0: No sufficient method found, skip this feature
% 1: SVM
% 2: KNN
% 3: Naive Bayes
binMethod = [0 0 2 0 0 0 2 0 0 0 2 1 2 0 2]; 

% Parameter set 
% 0: No method, vector of zeros 
% 1: Default SVM parameters
% k: Nearest neighbor parameter k 
binParams = [0 0 6 0 0 0 2 0 0 0 20 1 2 0 10]; 
%------------------------------------------------------------------------%
% Kernel ridge regression parameters: determined by experimentation
sigma = 2.^[12,8,7.5,8.5,13,7.75,7.5,7.5,9.5,9.3,6,7.3,6,5.3,9.6];
lambda = [0.05,0.01,0.1,0.2,1,0.3,0.1,0.1,0.04,0.04,2,0.5,0.01,0.01,0.01]; 

[kRows ~] = size(trainingData); 
kapRows = size(trainingData,1); 
kapCols = size(testData,1); 
ridgeKernel = zeros(kRows,kRows); 
predictKernel = zeros(kapRows,kapCols); 
%%
%------------------------------------------------------------------------%
% Train binary models and classify 
tBinStart = tic; 
for k=1:numBinFeatures
    % Rescale ratings to 0 or 1
    currFeature = binClassIdx(k);
    classes = testFeature(:,currFeature);
    idx = find(classes>0);
    classes(idx) = 1;
    
    switch binMethod(k)
        case 0 % No significant method found; set to zero
            fprintf('No significant correlation on feature %d; set to 0.\n',currFeature);
            predRatings(:,currFeature) = 0;
            
        case 1 % SVM
            fprintf('Computing SVM on Feature %d ...\n', currFeature);
            try % Error handling in the event SVM fails to converge
                SVMStruct = svmtrain(trainingData,classes);
                SVMpredictClasses = svmclassify(SVMStruct,testData);
                SVMpredictClasses(numPredTimeSteps+1:end,:) = [];
                predRatings(:,currFeature) = SVMpredictClasses;
            catch exception
                fprintf('SVM failed to converge on feature %d! \n',currFeature);
                predRatings(:,currFeature) = 0;
            end
            fprintf('Done.\n');
            
        case 2 % KNN
            fprintf('Computing KNN on Feature %d ...\n', currFeature);
            knnPredictClasses = knnclassify(testData,trainingData,classes,binParams(k));
            knnPredictClasses(numPredTimeSteps+1:end,:) = [];
            predRatings(:,currFeature) = knnPredictClasses;
            fprintf('Done.\n');
    end
end

tBinElapsed = toc(tBinStart); 
fprintf('Classified %d binary features in %.2f s.\n', ... 
        numBinFeatures,tBinElapsed);

%------------------------------------------------------------------------%
% Classify continuous features
tContStart = tic;
for k=1:numContFeatures
    currFeature = contClassIdx(k);
    Y = testFeature(:,currFeature);
    fprintf('Computing Kernel Ridge Regression on feature %d...\n',currFeature);
    
    for i=1:kRows
        for j=1:kRows
            ridgeKernel(i,j) = exp(-1/(2*sigma(k)^2)* ...
                (norm(trainingData(i,:) - ...
                trainingData(j,:))^2));
        end;
    end
    % Compute optimal alpha from training data
    alpha = (ridgeKernel+lambda(k)*eye(size(k)))\Y;
    
    % Compute ratings for new data = testData
    for i=1:kapRows
        for j=1:kapCols
            predictKernel(i,j) = exp(-1/(2*sigma(k)^2)*(norm(trainingData(i,:)-testData(j,:))^2));
        end
    end
    
    predictRatings = alpha'*predictKernel;
    predictRatings = predictRatings';
    predRatings(:,currFeature) = predictRatings;
    
    fprintf('Done.\n');
end
tContElapsed = toc(tContStart); 
fprintf('Classified %d continuous features in %.2f s.\n', ... 
    numContFeatures,tContElapsed);
fprintf('Total Time: %.2f s. \n', tBinElapsed+tContElapsed); 
%------------------------------------------------------------------------%
% Post-processing
idxHi = find(predRatings>1);
predRatings(idxHi) = 1; 
idxLo = find(predRatings<0); 
predRatings(idxLo) = 0; 
predRatings(:,7) = 0; % Food feature has slightly negative correlation
[rowidx colidx]= find(predRatings(:,30) >0);
predRatings(rowidx,1:29) = 0; 

%------------------------------------------------------------------------% 
% Score algorithm performance 
score = computeScore(trueRatings,predRatings); 
fprintf('Final Score: %f \n', score); 

if saveRes
    save('FMRIMindReadingSolution.mat','solution');
    fprintf('Results saved.\n'); 
end
