% score: Compute weighted correlation between true and estimated ratings 
% for each of 30 features.  
%-------------------------------------------------------------------------%
% inputs: 
%   -trueratings: ratings reference array, arranged as ratings (rows) by
%    features (columns)
%   -solution: candidate array of estimated ratings to be scored; 
%    dim. ratings x features 
%-------------------------------------------------------------------------%
% outputs:
%   -score: weighted sum of individual correlations
%-------------------------------------------------------------------------%

function score = computeScore(trueratings,solution) 

weights = [3*ones(1,13) ones(1,17)];
R = zeros(1,length(weights)); 

for k = 1:length(weights)
    C = cov(trueratings(:,k),solution(:,k),1); 
    if (C(1,1) ~= 0 && C(2,2) ~= 0) % neither vector is constant
        R(k) = C(1,2)/sqrt(C(1,1)*C(2,2));
    else 
        R(k) = 0;
    end
end
  
% Score is weighted sum of correlations  
score = sum(weights.*R); 

end
