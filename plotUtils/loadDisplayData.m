% Data loading and visualization script:
% loads feature set, MRI data, and predicted ratings, then displays useful
% visualizations for each 

clear
close all
baseDir = 'C:\Users\Chris\Dropbox\Documents\'; 
dataDir = [baseDir,'codeRepo\fMRIPrediction\data\'];
saveDir = [baseDir,'Blog\fMRI Prediction\'];

addpath(dataDir)
load ratings
load ROITimeseries
load originalProjReference
saveRes = 0; 
trueRatings = video2ratingshalf; 
predRatings = solution; 
labels = ratingsfeaturenames(:,2:end); 
binClassIdx = [14:1:27,30]; 
contClassIdx = [1:1:13,28,29]; 

% Coerce ratings values outside range to [0,1]
trueRatingsCopy = zeros(size(trueRatings));
trueRatingsCopy(:,contClassIdx) = trueRatings(:,contClassIdx);
binRatings = trueRatings(:,binClassIdx); 
binRatings(binRatings>0)=1; 
trueRatingsCopy(:,binClassIdx) = binRatings; 
trueRatings = trueRatingsCopy; 
%-------------------------------------------------------------------------%
% Concatenated ratings 
f1 = newFigPos(); 
imagesc([video1ratings' video2ratingshalf']); colormap hot

set(gca,'Ytick',1:1:30); 
hold on
plot([858 858],[0 31],':','Linewidth',4,'Color',[0 1 1])

set(gca,'YTicklabel',ratingsfeaturenames(2:end),'fontsize',16); 
xlabel('Time','FontSize',16,'FontName','Tw Cen MT');
colorbar

% Create arrow
annotation(f1,'arrow',[0.4848 0.4189],[0.9437 0.94377],'HeadStyle','plain');

% Create arrow
annotation(f1,'arrow',[0.7078 0.7747],[0.9436 0.9439],'HeadStyle','plain');

% Create textbox
annotation(f1,'textbox',[0.4306 0.9380 0.1056 0.04867],'String',{'Video 1'},...
    'FontSize',16,'FontName','Tw Cen MT','FitBoxToText','off', ...
    'EdgeColor','none','LineWidth',1.5);

% Create textbox
annotation(f1,'textbox', [0.7050 0.9542 0.09523 0.0324],'String',{'Video 2'},...
    'FontSize',16,'FontName','Tw Cen MT','FitBoxToText','off',...
    'EdgeColor','none','LineWidth',1.5);
set(f1,'position',get(0,'screensize'))

%-------------------------------------------------------------------------%
% fMRI Data
f2 = newFigPos(); 
imagesc([[ROIsLHvideo1 ROIsRHvideo1]',[ROIsLHvideo2 ROIsRHvideo2]']);
colormap hot
ylabel('Brain ROIs','FontSize',16,'FontName','Tw Cen MT'); 
xlabel('Time','FontSize',16,'FontName','Tw Cen MT');
hold on
plot([858 858],[0 301],':','Linewidth',4,'color',[0 1 1]);
colorbar
set(f2,'position',get(0,'screensize'))

%-------------------------------------------------------------------------%
% Plot individual ratings 
f3 = newFigPos(); 
featIdx = [1,5,9,13]; 
roiData = [ROIsLHvideo1 ROIsRHvideo1]';
ratingsData = [video1ratings' video2ratingshalf']; 

for k=1:length(featIdx)
    subplot(2,2,k);
    plot(ratingsData(featIdx(k),:),'ok');
    title(labels(featIdx(k)));
    axis tight
    set(gca,'xticklabel',[],'yticklabel',[]);
     grid on
end

% Create arrows
annotation(f3,'arrow',[0.2121 0.7722],[0.03400 0.03196]);
annotation(f3,'arrow',[0.1059 0.10698],[0.139 0.7716]);

% Create textboxes
annotation(f3,'textbox',[0.0075 0.4616 0.08676 0.03196],...
    'String',{'Rating'},'FontSize',16,'FontName','Tw Cen MT', ... 
    'FitBoxToText','off','EdgeColor','none');
annotation(f3,'textbox',[0.4827 0.05783 0.08677 0.0319],...
    'String',{'Time'},'FontSize',16,'FontName','Tw Cen MT', ... 
    'FitBoxToText','off','EdgeColor','none');

%-------------------------------------------------------------------------%
% Plot predicted and actual ratings 

f4 = newFigPos(); 

subplot(1,2,1);
imagesc(predRatings'); 
title('Predicted')
colormap hot
set(gca,'Ytick',1:1:30); 
hold on
set(gca,'YTicklabel',ratingsfeaturenames(2:end),'fontsize',16); 
xlabel('Time','FontSize',20,'FontName','Tw Cen MT');


subplot(1,2,2);
imagesc(trueRatings'); 
title('True Ratings')
colormap hot
set(gca,'Ytick',1:1:30); 
hold on
set(gca,'YTicklabel',ratingsfeaturenames(2:end),'fontsize',16); 
xlabel('Time','FontSize',20,'FontName','Tw Cen MT');
colorbar

%-------------------------------------------------------------------------%
% Plot individual feature predictions
featIdx = [25,26]; 
idx = 1:length(predRatings); 
f6 = newFigPos(); 

for k=1:length(featIdx)
    subplot(2,1,k);
    plot(idx,predRatings(:,featIdx(k)),'ob',idx,trueRatings(:,featIdx(k)),'or');
    title(labels(featIdx(k)));
    axis tight
    set(gca,'xticklabel',[],'yticklabel',[]);
     grid on
end
grid on
lgd = legend('Predicted','True');

set(lgd,'Position',[0.26453 0.46533 0.15081 0.09285],'fontsize',16);

% Create arrows
annotation(f6,'arrow',[0.2121 0.7722],[0.03400 0.03196]);
annotation(f6,'arrow',[0.1059 0.10698],[0.139 0.7716]);

% Create textboxes
annotation(f6,'textbox',[0.0075 0.4616 0.08676 0.03196],...
    'String',{'Rating'},'FontSize',16,'FontName','Tw Cen MT', ... 
    'FitBoxToText','off','EdgeColor','none');
annotation(f6,'textbox',[0.4827 0.05783 0.08677 0.0319],...
    'String',{'Time'},'FontSize',16,'FontName','Tw Cen MT', ... 
    'FitBoxToText','off','EdgeColor','none');

%-------------------------------------------------------------------------%
% Plot predicted feature correlations
f7 = newFigPos(); 
barh(featureCorrs,'k');
grid on
set(gca,'Ytick',1:1:30,'ydir','reverse'); 
axis tight
set(gca,'YTicklabel',ratingsfeaturenames(2:end),'fontsize',14); 
xlabel('Reference Correlation','FontSize',20,'FontName','Tw Cen MT');
xlim([0 1])


%-------------------------------------------------------------------------%
% Plot ratings against brain ROI with max correlation for feature
f8 = newFigPos(); 
featIdx = [1,5,9,13]; 
roiIdx = [39,65,84,size(ROIsRHvideo1,2)+47]; 
roiData = [ROIsRHvideo1 ROIsLHvideo1];
ratingsData = video1ratings; 


for k=1:length(featIdx)
    subplot(2,2,k);
    plot(roiData(:,roiIdx(k)),video1ratings(:,featIdx(k)),'ok');
    title(labels(featIdx(k)));
    axis tight
    set(gca,'xticklabel',[],'yticklabel',[]);
     grid on
end

% Create arrows
annotation(f8,'arrow',[0.2121 0.7722],[0.03400 0.03196]);
annotation(f8,'arrow',[0.1059 0.10698],[0.139 0.7716]);

% Create textboxes
annotation(f8,'textbox',[0.0075 0.4616 0.08676 0.03196],...
    'String',{'Rating'},'FontSize',16,'FontName','Tw Cen MT', ... 
    'FitBoxToText','off','EdgeColor','none');
annotation(f8,'textbox',[0.3555 0.0173355 0.44006 0.06122],...
    'String',{'Max Corr. Brain ROI Activity'},'FontSize',16,'FontName','Tw Cen MT', ... 
    'FitBoxToText','off','EdgeColor','none');


if saveRes
    saveas(f1,[saveDir,'ratingsAll'],'jpg')
    saveas(f2,[saveDir,'fMRI_ROIs'],'jpg')
    saveas(f3,[saveDir,'contRatingsSubset'],'jpg')
    saveas(f4,[saveDir,'trueAndPredRatingsAllVideo2'],'jpg')
    saveas(f6,[saveDir,'predBinRatingsSubset_bad'],'jpg')
    saveas(f7,[saveDir,'predFeatCorrelations'],'jpg')
    saveas(f8,[saveDir,'contRatingsSubset_pkROI_CORR'],'jpg')
end
