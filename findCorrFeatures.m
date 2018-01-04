% Returns vector of correlations between testFeature and testROI, maximum 
% correlation maxCorr, and index of elements in testROI whose absolute 
% correlation is above parameter thresh*maxCorr.  

% ex: 
% testFeature: video1ratings; 
% testROI: [ROIsLHvideo1 ROISLHVideo2]

function [featureCorrs,maxCorr,sigIdx] = findCorrFeatures(testFeature,testROI,thresh) 

featureCorrs = zeros(1,size(testROI,2)); 
cols = size(testROI,2); 
for k = 1:cols
    featureCorrs(k) = corr(testFeature,testROI(:,k)); 
end
maxCorr = max(featureCorrs(:)); 
sigIdx = find(abs(featureCorrs)>=thresh*maxCorr); 

end


