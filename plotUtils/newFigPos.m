% newFigPos: Generates new figure that is automatically tiled with 
% respect to all currently open figure windows

% - Inputs - 
% posVec: position vector in format: 
% [fixed left offset, ... 
% left/bottom offset multiplier, ... 
% figure bottom offset, ... 
% figure width, ... 
% figure height]

% - Outputs - 
% h: handle to figure window 

function h = newFigPos(posVec)
h = figure; 

if nargin < 1
      posVec = [25 15 400 700 600];
end
nFigs = length(findall(0,'type','figure'))+1; 
set(h,'Position',[posVec(1)+nFigs*posVec(2) posVec(3)-nFigs*posVec(2), ... 
                 posVec(4) posVec(5)]);

