# fMRIPrediction
Training model for discrete and continuous feature ratings prediction from fMRI brain imaging data. 
Requires Bioinformatics toolbox (k-nearest neighbors) and Statistics and Machine learning toolbox (SVM/Naive Bayes).
Includes raw data sets and final predicted solutions in \data. 

Main script:   
-ratingsMRIdata_trainModel: uses best results determined from experimental scratchpad.  
Support:  
-loadDisplayData: visualization and presentation of raw data and computed results.   

Scratchpad experimental code for generating predictions with fixed sets of parameters:   
-binClassifiers: binary classifiers - trains and scores knn, Naive Bayes, and SVM models with fixed set of parameters   
-contClassifiers_singleParam:  continuous classifiers - trains and scores regression methods with fixed set of parameters   
for Gaussian kernel width and ridge regression penalty factor.   
-contClassifiers_paramSet: continuous classifiers - same as _singleParam, but sweeps over range of parameters for   
to determine best performance.   
