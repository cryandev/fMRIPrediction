% Experimentation scratchpad for binary classifiers (features 14-27,30) 
% Adjustment of parameters for native implementations of SVM, KNN, 
% and Naive Bayes algorithms 

close all
% Define global feature index 
featureIdx = 1:1:30; 
testFeatureCorrs = cell(1,30); 
sigIdx = cell(1,30); 

% Define binary feature indices 
binClassIdx = [14:1:27,30]; 
% Pre-allocate arrays 
SVMsolution = zeros(434,30); 
NBsolution = zeros(434,30); 
knnSolution = zeros(434,30); 
SVMcorr = zeros(1,30); 
NBcorr = zeros(1,30); 
knnCorr = zeros(1,30); 

trainingData = [ROIsLHvideo1 ROIsRHvideo1]; 
testData = [ROIsLHvideo2 ROIsRHvideo2]; 
trueClasses = video2ratingshalf; 

numNeighbors = 100; 

tStart = tic; 
for k = 1:length(binClassIdx)

    currClass = binClassIdx(k); 
    classes = video1ratings(:,currClass);   
    idx = find(classes>0); 
    classes(idx) = 1; 
    sigma = 2^1; 
   
    fprintf('Computing SVM on Feature %d ...\n', currClass);     
    try 
    SVMStruct = svmtrain(trainingData,classes,'Kernel_function','rbf','rbf_sigma',sigma); 
    SVMStruct = svmtrain(trainingData,classes); 
    SVMpredictClasses = svmclassify(SVMStruct,testData); 
    SVMpredictClasses(435:end,:) = []; 
    
    catch exception 
        fprintf('SVM failed to converge on feature %d! \n',currClass); 
        SVMpredictClasses = zeros(434,1); 
    end; 
    fprintf('Done.\n'); 

    fprintf('Computing NaiveBayes on Feature %d ...\n', currClass); 
    NB = NaiveBayes.fit(trainingData,classes); 
    BayesPredictClasses = NB.predict(testData); 
    BayesPredictClasses(435:end,:) = []; 

    fprintf('Done.\n'); 
    
    fprintf('Computing KNN on Feature %d ...\n', currClass); 
    knnPredictClasses = knnclassify(testData,trainingData,classes,numNeighbors); 
    knnPredictClasses(435:end,:) = []; 

    fprintf('Done.\n'); 
    
  
SVMsolution(:,currClass) = SVMpredictClasses; 
NBsolution(:,currClass) = BayesPredictClasses; 
knnSolution(:,currClass) = knnPredictClasses; 

SVMcorr(currClass) = corr(trueClasses(:,currClass),SVMsolution(:,currClass));  
NBcorr(currClass) = corr(trueClasses(:,currClass),NBsolution(:,currClass));  
knnCorr(currClass) = corr(trueClasses(:,currClass),knnSolution(:,currClass));  

end;

tElapsed = toc(tStart); 
fprintf('Classified %d features in %.2f s.\n', length(binClassIdx),tElapsed); 

SVMcorr(find(isnan(SVMcorr))) = 0; 
NBcorr(find(isnan(NBcorr))) = 0; 
knnCorr(find(isnan(knnCorr))) = 0; 

SVMscore = computeScore(trueClasses,SVMsolution); 
NBscore = computeScore(trueClasses,NBsolution); 
knnScore = computeScore(trueClasses,knnSolution); 
fprintf('Sum SVM Correlations: %f \n', sum(SVMcorr(:)')); 

% Plot binary classifier metrics
fullScreen = get(0,'Screensize'); 
figure
set(gcf,'Position',fullScreen); 
h1 = subplot(2,2,1); imagesc(video1ratings(430:end,:)); a1 = gca; 
h2 = subplot(2,2,2); imagesc(SVMsolution); 
h3 = subplot(2,2,3); imagesc(NBsolution); 
h4 = subplot(2,2,4); imagesc(knnSolution);

title(h1,'True Ratings','FontWeight','Bold'); 
title(h2, ['SVM Predicted Ratings: ',num2str(SVMscore)],'FontWeight','Bold'); 
title(h3, ['KNN Predicted Ratings: ',num2str(NBscore)],'FontWeight','Bold'); 
title(h4, ['NaiveBayes Predicted Ratings: ',num2str(knnScore)],'FontWeight','Bold'); 
xlabel(a1,'Category'); 
ylabel(a1,'Time'); 
set([h1 h2 h3 h4],'Xtick',1:1:30); 
set([h1 h2 h3 h4],'XGrid','on'); 
colormap hot

figure
plot(featureIdx,SVMcorr,'k',featureIdx,NBcorr,'--b',featureIdx,knnCorr,'-r'); 
legend('SVM Corr','NB Corr','KNN Corr'); 
set(gca,'Xtick',featureIdx); 
set(gca,'XGrid','on'); 
xlabel('Feature'); ylabel('Correlation'); 
