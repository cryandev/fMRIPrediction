% Generate default settings for all plots     
close all
width = 7;     % Width in inches
height = 6;    % Height in inches
alw = 0.75;    % AxesLineWidth
fs = 16;       % FontSize l
% fontname = 'Times';
fontname = 'Tw Cen MT'; % Font name
lw = 1.5;      % LineWidth
msz = 5;       % MarkerSize

% Set the default Size for display
% [left bottom width height]
defpos = get(0,'screensize');

set(0,'defaultLineLineWidth',lw);   
set(0,'defaultLineMarkerSize',msz); 
set(0,'defaultLineLineWidth',lw);   
set(0,'defaultLineMarkerSize',msz); 
set(0,'defaultAxesFontSize',fs);   
set(0,'defaultAxesLineWidth',alw);
set(0,'defaultAxesFontName',fontname);
set(0,'defaultFigurePaperPositionMode','auto');
set(0, 'defaultFigurePaperPosition', [defpos(4)/2 defpos(3)/8 700 600]);
clear width height alw fs fontname lw msz pos defpos

%%
Set the defaults for saving/printing to a file
set(0,'defaultFigureInvertHardcopy','on'); 
set(0,'defaultFigurePaperUnits','inches'); 
defsize = get(gcf, 'PaperSize');
left = (defsize(1)- width)/2;
bottom = (defsize(2)- height)/2;
defsize = [left, bottom, width, height];
set(0, 'defaultFigurePaperPosition', defsize);
