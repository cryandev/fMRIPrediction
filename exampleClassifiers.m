% Test examples: experimentations with SVM utilities 

% Create training data from Gaussian at (0,0) and (2,2) and plot
training_data = [randn(2,50) 2*ones(2,50)+ randn(2,50)];
classes = [ones(1,50) 2*ones(1,50)]';
figure
scatter(training_data(1,:), training_data(2,:), 30, classes, 'filled'); 

% Create new data from same distribution
test_data = [randn(2,5) 2*ones(2,5)+ randn(2,5)];
test_classes = [ones(1,5) 2*ones(1,5)]';
hold on
scatter(test_data(1,:), test_data(2,:), 90, test_classes, 'filled');

% Use SVM to partition data
SVMstruct = svmtrain(training_data, classes, 'Kernel_Function', 'linear', ... 
'showplot', true);
est_classes = svmclassify(SVMstruct, test_data'); 

% Use knn to partition data
knnclassify(test_data', training_data', classes, 3); 

% Use Naive Bayes to classify data
O1 = NaiveBayes.fit(training_data', classes); 
C1 = O1.predict(test_data')


%% Experimentation with real data 
training_data = [ROIsLHvideo1 ROIsRHvideo1]; 
test_data = [ROIsLHvideo2 ROIsRHvideo2]; 
trueClasses = video2ratingshalf; 
k = 1; 
currClass = BinClassIdx(k); 
classes = video1ratings(:,currClass);   
idx = find(classes>0); 
classes(idx) = 1;
SVMStruct = svmtrain(training_data,classes); 
SVMpredictClasses = svmclassify(SVMStruct,test_data); 

NB = NaiveBayes.fit(training_data,classes); 
BayesPredictClasses = NB.predict(test_data); 

knnPredictClasses = knnclassify(test_data,training_data,classes,numNeighbors); 